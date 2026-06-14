# MODIFICATIONS.md — Tactile-Video Generation Fine-tuning for Show-o2

## Overview

This document describes all code changes made to fine-tune Show-o2 into a **haptic video understanding & generation model** that takes physical text descriptions + visual video frames as input and generates tactile video frames as output.

**Core design principle**: Tactile video is treated as **another type of visual modality** — no new tokens, no new embedders, no model architecture changes. All changes are at the **data layer** and **sequence formatting** level.

### Sequence Structure

```
[BOS] {Physical Text: "Contact: True, Material: Rough Wood"}
[BOV] {Visual Video Latents (condition, clean, t=1.0)} [EOV]
[BOV] {Tactile Video Latents (target, noised, flow loss)} [EOV]
[EOS]
```

---

## Files Created (8 new files)

### 1. `datasets/tactile_visual_dataset.py` — **NEW**

**Purpose**: Dataset class that loads tactile-visual paired data from disk.

**What it does**:
- Reads CSV metadata (`contact_indoor_list_tvl.csv`) to get object names, text descriptions, and train/test splits
- For each object, loads 2 video sequences:
  - `img_gelsight/` → visual conditioning video frames (640×480 original)
  - `gelsight/` → target tactile video frames (240×320 original)
- Uniformly samples `num_frames` (default=5) from each video
- Applies image transform (resize to 432×432, center crop, normalize)
- Constructs the unified token sequence via `format_sequence_tactile_gen()`
- Returns `data_type = "tactile_visual_data"` for special handling in the training loop

**Key design**: `images` tensor shape is `(2, num_frames, C, H, W)`:
- `images[0]` = visual conditioning video
- `images[1]` = tactile target video

**Why this approach**: Follows the exact pattern of `VISTDataset` but specialized for two-video (visual+tactile) pairs. The interleaved data pattern from `train_mixed_modality_simple.py` is reused.

---

### 2. `train_tactile_stage_one.py` — **NEW**

**Purpose**: Stage 1 training script — warms up fusion projection and diffusion head for tactile domain.

**What it does**:
- Loads pre-trained Show-o2 (1.5B) model
- Freezes: `image_embedder_und`, `und_trans`, `showo` (LLM backbone), `position_embedding`
- Trains: `fusion_proj`, `diffusion_head_a/b`, `time_embed`, `image_embedder_gen`
- `prepare_latents_and_labels()` implements the tactile-specific logic:
  1. VAE-encodes all frames independently: `(B*2*T, C, H, W)` → `(B*2*T, 16, Hv, Wv)`
  2. Applies transport noising to each frame
  3. **Sets visual video frames to clean**: `t=1.0`, `xt = original latents`
  4. **Masks flow loss for visual segment**: sets `image_masks[vis_sid:vis_sid+vis_len] = 0`
  5. **Keeps tactile frames noised**: flow loss computed only on tactile segment
  6. Reshapes to `(B*2, 16, T, Hv, Wv)` for the model's per-video 5D format

**Based on**: `train_mixed_modality_simple.py` (interleaved multi-image pattern)

**Why this approach**: The interleaved data pattern already supports selective noising of images. We extend it so that visual video (position 0) is always clean and tactile video (position 1) is always noised, rather than random selection.

---

### 3. `train_tactile_stage_two.py` — **NEW**

**Purpose**: Stage 2 training script — full model fine-tuning with differentiated learning rates.

**What it does**:
- Loads Stage 1 checkpoint
- Unfreezes all parameters
- Creates 3 optimizer parameter groups with different learning rates:
  - **Vision Encoder** (`image_embedder_und`, `und_trans`, `position_embedding`): `lr = 2e-6` — preserve pre-trained semantic features
  - **Projector + Diffusion** (`fusion_proj`, `diffusion_head`, `time_embed`, `image_embedder_gen`): `lr = 1e-5` — primary adaptation targets
  - **LLM Backbone** (`showo`): `lr = 2e-6` — gentle fine-tuning
- Uses cosine LR schedule with 3% warmup
- Same `prepare_latents_and_labels()` logic as Stage 1

**Why differentiated LRs**: The semantic layers (SigLIP-derived) already produce good features — we just need gentle nudges. The fusion+diffusion layers learned tactile domain statistics in Stage 1 and need continued refinement. The LLM gets a very low LR to preserve its language understanding while slightly adapting to tactile context.

---

### 4. `inference_tactile_video.py` — **NEW**

**Purpose**: Inference script for tactile video generation.

**What it does**:
1. Loads a fine-tuned model checkpoint
2. Accepts text prompt + directory of visual video PNG frames
3. VAE-encodes visual frames (as clean condition)
4. Initializes random noise for tactile latent
5. Constructs the 3-modality sequence: `text + [BOV]visual[EOV] + [BOV]noise[EOV]`
6. Uses Flow-Matching ODE solver (Euler method) to iteratively denoise the tactile latent
7. VAE-decodes the result into tactile video frames
8. Saves as MP4 or PNG sequence

**Key flags**:
- `only_denoise_last_image=True`: Only the tactile segment (last in modality_positions) gets denoised
- `guidance_scale`: CFG scale for text conditioning (default 5.0)
- The visual video segment has `t=1.0` (clean) throughout denoising

---

### 5. `configs/showo2_1.5b_tactile_stage_one.yaml` — **NEW**

**Purpose**: Stage 1 configuration with all hyperparameters externalized.

Key parameters and their selection rationale:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_seq_length` | 8192 | Text(~400) + 2 videos (2×5×729=7290) + special tokens(6) ≈ 7700, rounded to 8192 |
| `resolution` | 432 | Matches Show-o2 pre-training resolution; compatible with 8× VAE + 2× patch |
| `num_frames` | 5 | Trade-off: enough temporal info for tactile texture dynamics; 5 frames of 432×432 = 3645 tokens per video |
| `learning_rate` | 1e-4 | Standard for Stage 1 projector warm-up (matches original Show-o2 Stage 1) |
| `batch_size_tactile` | 2 | Each sample = 2 videos × 5 frames = 10 images; 2 samples/GPU fits 24GB VRAM |
| `max_train_steps` | 30,000 | ~300 epochs over 99 samples at batch 2; sufficient for domain adaptation |
| `ntp_coeff` | 0.0 | No text generation loss — only flow matching on tactile |
| `flow_coeff` | 1.0 | Full weight on flow matching loss |
| `cond_dropout_prob` | 0.1 | 10% text dropout for CFG training |
| `snr_type` | "lognorm" | Log-normal SNR sampling for better noise level coverage |
| `warmup_steps` | 500 | Short warmup to stabilize initial gradients on new domain |
| `seed` | 42 | Fixed for reproducibility |

---

### 6. `configs/showo2_1.5b_tactile_stage_two.yaml` — **NEW**

**Purpose**: Stage 2 configuration with differentiated LRs.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `learning_rate_ve` | 2e-6 | SigLIP semantic layers: ~50× lower than Stage 1 to preserve pre-trained features |
| `learning_rate_proj` | 1e-5 | Fusion + diffusion: primary adaptation targets, moderate LR |
| `learning_rate_showo` | 2e-6 | LLM backbone: minimal change to preserve language understanding |
| `max_train_steps` | 10,000 | ~100 epochs; smaller than Stage 1 since weights are already in the right basin |
| `lr_scheduler` | cosine | Smooth decay for convergence; 3% warmup |
| `frozen_params` | `[]` (empty) | All parameters trainable |

---

### 7. `run_tactile_stage_one.sh` — **NEW**

Launch script for Stage 1 with CLI-overridable hyperparameters.

### 8. `run_tactile_stage_two.sh` — **NEW**

Launch script for Stage 2 with CLI-overridable hyperparameters.

---

## Files Modified (3 existing files)

### 1. `datasets/utils.py` — **MODIFIED**

**Change**: Added `format_sequence_tactile_gen()` function (lines inserted before `resize_crop`).

**What it does**: Builds the 3-modality token sequence:
```python
[BOS] + text_tokens + [BOV] + [vid_pad]*N_vis + [EOV] + [BOV] + [vid_pad]*N_tac + [EOV] + [EOS] + [PAD]...
```

Returns `modality_positions` as a `(2, 2)` tensor:
```python
[[visual_offset, num_visual_tokens],   # visual video segment
 [tactile_offset, num_tactile_tokens]] # tactile video segment
```

**Why**: The existing `format_interleaved_sequence` is designed for alternating text+image sequences (VIST storytelling). Our tactile task has a different pattern: text → visual video → tactile video with specific `[BOV]...[EOV]` delimiters for video segments. The new function is cleaner and more explicit for this use case.

**Key design**: All labels are `-100` (no NTP loss). Since we only train flow matching on the tactile segment, there's no benefit to computing cross-entropy on text tokens. The `ntp_coeff=0.0` in config confirms this.

---

### 2. `models/misc.py` — **MODIFIED**

**Change**: Added `prepare_tactile_gen_input()` function (lines inserted before `prepare_gen_input`).

**What it does**: Prepares token sequences for inference:
- Builds the 3-modality sequence: `[BOS]text[BOV]visual_pads[EOV][BOV]tactile_pads[EOV]`
- Creates a null-text variant for CFG (classifier-free guidance)
- Returns `(batch_text_tokens, batch_text_tokens_null, batch_modality_positions, batch_modality_positions_null)`

**Why**: The existing `prepare_gen_input` only handles single-image generation. For tactile generation, we need two visual regions (condition visual + target tactile) with correct offsets. This function mirrors `prepare_mixed_modal_gen_input` but specialized for the tactile sequence layout.

---

### 3. `datasets/__init__.py` — **MODIFIED**

**Change**: Added import line:
```python
from .tactile_visual_dataset import TactileVisualDataset
```

**Why**: Makes the new dataset class importable from `datasets` package, consistent with existing `VISTDataset`.

---

## Files NOT Modified (Zero Changes Required)

These files need **zero changes** because the "tactile = visual" design principle is fully supported by the existing architecture:

| File | Why No Changes Needed |
|------|----------------------|
| `models/modeling_showo2_qwen2_5.py` | `forward()` already handles multi-segment visual inputs via `modality_positions` loop; 5D latents `(B, C, T, H, W)` for video already supported; `only_denoise_last_image` flag already exists |
| `models/omni_attention.py` | `omni_attn_mask_naive()` already processes multiple modality regions per sample via its inner loop over `modality_batch` |
| `models/wan21_vae.py` | Frozen VAE shared for both visual and tactile encoding; no changes needed |
| `models/modules.py` | ModulatedAttentionBlock + FinalLayer operate on per-position basis; modality-agnostic |
| `models/modeling_siglip.py` | SigLIP semantic layers shared for all visual inputs |
| `transport/` | Flow-matching framework completely reused; `transport.sample()` works with any tensor shape |
| `datasets/mixed_dataloader.py` | Already handles single-loader case with `max_size_cycle` mode |

---

## Parameter Choice Rationale (Detailed)

### Sequence Length: 8192

```
[BOS]              = 1 token
Text               = ~200-400 tokens (typical physical descriptions are short)
[BOV]              = 1 token
Visual pads        = 5 frames × 729 tokens/frame = 3645 tokens
[EOV]              = 1 token
[BOV]              = 1 token
Tactile pads       = 5 frames × 729 tokens/frame = 3645 tokens
[EOV]              = 1 token
[EOS]              = 1 token
─────────────────────────────────────────────────
Total              ≈ 7500 tokens (text-dependent)
+ Time embeddings  = +2 tokens (one per video segment, when add_time_embeds=True)
+ Padding          = to 8192
```

### Resolution: 432×432

- Show-o2 was pre-trained at 432×432 resolution
- VAE compression: 432/8 = 54 spatial latent → 54/2 (patch_size) = 27 tokens per spatial dim → 27×27 = 729 tokens/frame
- Original tactile frames (240×320) and visual frames (640×480) are both resized to 432×432 with center crop
- Trade-off: higher resolution would give finer texture detail but longer sequences

### Number of Frames: 5

- Tactile texture dynamics are relatively slow — 5 frames at ~1fps captures contact deformation well
- 5 frames × 729 tokens = 3645 tokens per video, which fits within 8192 sequence budget
- Can be increased to 9 or 17 (like the paper's 17-frame video) if more temporal resolution is needed

### Learning Rates (Stage 1)

- **1e-4 for fusion_proj + diffusion_head**: Same as original Show-o2 Stage 1. These weights are randomly initialized (Xavier) and need significant updates. Higher LR is safe because the LLM is frozen.
- **Frozen LLM**: The Qwen2.5 backbone's language understanding doesn't need adaptation for tactile data.

### Learning Rates (Stage 2)

- **2e-6 for semantic layers**: 50× lower than Stage 1. SigLIP features are pre-trained on large-scale vision data and are high-quality. Over-training on 99 tactile samples would destroy this knowledge.
- **1e-5 for fusion+diffusion**: These adapt the most in Stage 2 since they directly process tactile features.
- **2e-6 for LLM**: Minimal change — the model mainly needs to learn to attend to visual context for tactile generation, not to change its language model.
- **Cosine schedule**: Smooth decay avoids oscillation at convergence.

### SNR Type: "lognorm"

- Log-normal distribution of noise levels: more samples at intermediate noise levels where the model learns the most
- Better than uniform sampling for small datasets

### Training Steps

- **Stage 1 (30K steps)**: At batch 2, that's 60K samples → ~600 epochs over 99 objects. For learning a new visual domain (tactile vs. natural images), this is a reasonable warm-up.
- **Stage 2 (10K steps)**: At batch 2, that's 20K samples → ~200 epochs. Fine-tuning with low LR requires fewer steps to converge.

---

## Training Flow Summary

```
Stage 1 (30K steps):
  Frozen: image_embedder_und, und_trans, showo, position_embedding
  Trained: fusion_proj, diffusion_head, time_embed, image_embedder_gen
  LR: 1e-4 (constant with 500-step warmup)
  Loss: Flow matching only (ntp_coeff=0.0)

Stage 2 (10K steps):
  All parameters unfrozen
  LR: 2e-6 (semantic), 1e-5 (fusion+diffusion), 2e-6 (LLM)
  LR schedule: Cosine with 3% warmup
  Loss: Flow matching only (ntp_coeff=0.0)
  Resume from: Stage 1 final checkpoint
```

---

## Inference Flow Summary

```
Input: Text prompt + visual video frames (PNG directory)

1. Tokenize text → text_tokens
2. VAE encode visual frames → visual_latents (clean, t=1.0)
3. Initialize random noise → tactile_latents
4. Build sequence: [BOS]text[BOV]visual[EOV][BOV]noise[EOV]
5. Build omni_attn_mask (causal for text, bidirectional within each video segment)
6. ODE denoise (Euler, 50 steps) → tactile_latents (cleaned)
7. VAE decode → tactile video frames
8. Save as MP4/PNG sequence
```

---

## Quick Start

```bash
# Stage 1: Adapt fusion + diffusion to tactile domain
bash run_tactile_stage_one.sh

# Stage 2: Full model fine-tuning (resume from Stage 1)
bash run_tactile_stage_two.sh

# Inference: Generate tactile video from text + visual video
python inference_tactile_video.py \
    --config configs/showo2_1.5b_tactile_stage_two.yaml \
    --checkpoint outputs/showo2-1.5b-tactile-stage-2/checkpoint-final/unwrapped_model \
    --text "Contact: True, Material: Rough Wood, it is made of rubber." \
    --visual_video_dir /path/to/test_object/img_gelsight/ \
    --output_path output_tactile.mp4 \
    --num_frames 5 \
    --guidance_scale 5.0
```
