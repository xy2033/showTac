# coding=utf-8
# Copyright 2025 NUS Show Lab.
#
# Stage 2 Tactile Video Generation & QA Inference.
#
# Supports:
#   1. Pure generation: text + visual video → tactile video (same as Stage 1)
#   2. QA generation: question + visual video → tactile video + answer text
#
# Usage:
#   # Single pure generation
#   python inference_tactile_video_stage2.py \
#       --stage2_checkpoint outputs/showo2-1.5b-tactile-stage-2-qa/checkpoint-3000/unwrapped_model \
#       --vae_path /path/to/Wan2.1_VAE.pth \
#       --llm_path /path/to/Qwen2.5-1.5B-Instruct \
#       --showo_path /path/to/show-o2-1.5B \
#       --siglip_path /path/to/siglip-so400m-patch14-384 \
#       --text "Contact: True, Material: rubber." \
#       --visual_video_dir /path/to/img_gelsight/ \
#       --output_path output_tactile.mp4
#
#   # Batch test (pure generation)
#   python inference_tactile_video_stage2.py \
#       --batch_test \
#       --stage2_checkpoint ... --vae_path ... --llm_path ... --showo_path ... --siglip_path ... \
#       --tactile_data_root /path/to/data --tactile_csv_path /path/to/contact.csv \
#       --output_dir /path/to/output
#
#   # QA batch test
#   python inference_tactile_video_stage2.py \
#       --batch_test --qa_mode \
#       --stage2_checkpoint ... --tactile_data_root ... --tactile_qa_csv_path tac_QA/tactile_qa_pairs.csv \
#       --output_dir /path/to/output
# =============================================================================

import argparse
import glob
import json
import os
import random
from contextlib import nullcontext

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from datasets import TactileVisualDataset
from datasets.tactile_qa_dataset import TactileQADataset
from datasets.utils import format_sequence_tactile_gen, format_sequence_tactile_qa
from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
from models.misc import get_text_tokenizer, interpolate_pos_encoding
from utils import path_to_llm_name, denorm_vid
from transport import Sampler, create_transport

# ==============================================================================
# Defaults (match training config)
# ==============================================================================
DEFAULT_RESOLUTION = 432
DEFAULT_NUM_FRAMES = 5
DEFAULT_LATENT_H = 27
DEFAULT_LATENT_W = 27
DEFAULT_TOKENS_PER_FRAME = 729
DEFAULT_SEQ_LEN = 8192


def autocast_context(device, dtype):
    device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    if device_type == "cuda" and dtype in (torch.bfloat16, torch.float16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def resolve_video_token_layout(model, num_frames):
    patch_size = int(getattr(model.config, "patch_size", 2))
    latent_h = int(getattr(model.config, "image_latent_height", DEFAULT_LATENT_H))
    latent_w = int(getattr(model.config, "image_latent_width", DEFAULT_LATENT_W))
    latent_tokens = num_frames * latent_h * latent_w
    add_time_token = bool(getattr(model.config, "add_time_embeds", False))
    segment_tokens = latent_tokens + int(add_time_token)
    max_text_len = DEFAULT_SEQ_LEN - 2 * segment_tokens - 6
    return latent_tokens, segment_tokens, max_text_len, add_time_token


def resolve_checkpoint_dir(checkpoint_path):
    if os.path.exists(os.path.join(checkpoint_path, "config.json")):
        return checkpoint_path
    unwrapped = os.path.join(checkpoint_path, "unwrapped_model")
    if os.path.exists(os.path.join(unwrapped, "config.json")):
        return unwrapped
    return checkpoint_path


def resolve_weight_file(model_path):
    if os.path.isdir(model_path):
        candidate_dirs = [model_path]
        unwrapped_dir = os.path.join(model_path, "unwrapped_model")
        if os.path.isdir(unwrapped_dir):
            candidate_dirs.append(unwrapped_dir)
        if os.path.basename(os.path.normpath(model_path)) == "unwrapped_model":
            parent_dir = os.path.dirname(os.path.normpath(model_path))
            if os.path.isdir(parent_dir):
                candidate_dirs.append(parent_dir)

        # Prefer single-file checkpoints, then fall back to sharded index files
        # (save_pretrained shards once the state dict exceeds max_shard_size).
        known_weight_names = (
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.bin.index.json",
            "model.safetensors.index.json",
            "diffusion_pytorch_model.bin.index.json",
            "diffusion_pytorch_model.safetensors.index.json",
        )
        for candidate_dir in candidate_dirs:
            for filename in known_weight_names:
                weight_file = os.path.join(candidate_dir, filename)
                if os.path.exists(weight_file):
                    return weight_file

        # Some checkpoints are saved with variants or custom prefixes, e.g.
        # model.fp16.safetensors.index.json or pytorch_model-00001-of-00002.bin.
        # Discover those instead of treating a valid sharded checkpoint as empty.
        for candidate_dir in candidate_dirs:
            for index_file in sorted(glob.glob(os.path.join(candidate_dir, "*.index.json"))):
                try:
                    with open(index_file, "r", encoding="utf-8") as f:
                        if "weight_map" in json.load(f):
                            return index_file
                except (OSError, json.JSONDecodeError):
                    continue

        shard_patterns = (
            "*-*-of-*.safetensors",
            "*-*-of-*.bin",
            "*.safetensors",
            "*.bin",
        )
        for candidate_dir in candidate_dirs:
            for pattern in shard_patterns:
                shard_files = sorted(glob.glob(os.path.join(candidate_dir, pattern)))
                if shard_files:
                    return shard_files[0]
    elif os.path.exists(model_path):
        return model_path
    return None


def describe_checkpoint_dir(model_path):
    if not os.path.isdir(model_path):
        return f"{model_path} is not a directory"
    entries = sorted(os.listdir(model_path))
    preview = ", ".join(entries[:30])
    if len(entries) > 30:
        preview += f", ... ({len(entries)} files total)"
    return preview


def load_single_weight_file(weight_file):
    if weight_file.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(weight_file, device="cpu")
    return torch.load(weight_file, map_location="cpu")


def load_checkpoint_state_dict(weight_file):
    # Sharded checkpoint: merge every shard listed in the index's weight_map.
    if weight_file.endswith(".index.json"):
        shard_dir = os.path.dirname(weight_file)
        with open(weight_file, "r", encoding="utf-8") as index_file:
            index = json.load(index_file)
        merged = {}
        for shard_name in sorted(set(index["weight_map"].values())):
            merged.update(load_single_weight_file(os.path.join(shard_dir, shard_name)))
        return merged

    # Fallback for directories where only shard files are present and no index
    # was found. This path is entered only once from the first shard.
    basename = os.path.basename(weight_file)
    if "-00001-of-" in basename:
        shard_prefix = basename.split("-00001-of-")[0]
        shard_match = sorted(glob.glob(os.path.join(os.path.dirname(weight_file), shard_prefix + "-*-of-*")))
        if len(shard_match) > 1:
            merged = {}
            for shard_name in shard_match:
                merged.update(load_single_weight_file(shard_name))
            return merged

    return load_single_weight_file(weight_file)


def load_model(args, device, weight_type):
    """Load the full Show-o2 model with Stage 2 fine-tuned weights."""
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        args.llm_path, add_showo_tokens=True, return_showo_token_ids=True,
        llm_name=path_to_llm_name[args.llm_path],
    )

    stage2_dir = resolve_checkpoint_dir(args.stage2_checkpoint)

    # Do not call from_pretrained() here: the saved config may contain HF Hub
    # repo ids such as Qwen/Qwen2.5-1.5B-Instruct, which fail on offline clusters.
    # Build the architecture with CLI-provided local paths, then load weights.
    config_path = os.path.join(stage2_dir, "config.json")
    if os.path.exists(config_path):
        print(f"Loading Stage 2 config from {config_path}...")
        with open(config_path, "r", encoding="utf-8") as config_file:
            cfg = json.load(config_file)
    else:
        print(f"[WARN] config.json not found in {stage2_dir}; using default 1.5B tactile config.")
        cfg = dict(
            model_name="Showo2", llm_model_path=args.llm_path,
            hidden_size=1536, image_latent_dim=16,
            image_latent_height=DEFAULT_LATENT_H, image_latent_width=DEFAULT_LATENT_W,
            patch_size=2, num_diffusion_layers=10, clip_latent_dim=1152,
            add_qk_norm=True, add_time_embeds=True,
        )

    cfg["llm_model_path"] = args.llm_path
    cfg["llm_vocab_size"] = len(text_tokenizer)
    cfg["load_from_showo"] = True
    if not os.path.isdir(args.llm_path):
        print(f"  [WARN] --llm_path '{args.llm_path}' does not exist or is not a directory")

    if args.siglip_path:
        cfg["clip_pretrained_model_path"] = args.siglip_path
    else:
        old_siglip_path = cfg.get("clip_pretrained_model_path")
        if not old_siglip_path or not os.path.isdir(old_siglip_path):
            raise ValueError(
                f"Config contains clip_pretrained_model_path='{old_siglip_path}', "
                "which is not available locally. Pass --siglip_path with the local "
                "SigLIP directory, e.g. --siglip_path /defaultShare/models/siglip-so400m-patch14-384"
            )

    print(f"Building Show-o2 with local LLM path: {args.llm_path}")
    model = Showo2Qwen2_5(**cfg).to(device)

    if args.showo_path:
        base_weight_file = resolve_weight_file(args.showo_path)
        if base_weight_file is None:
            raise FileNotFoundError(f"Base Show-o2 weights not found in --showo_path={args.showo_path}")
        print(f"Loading base Show-o2 weights from {base_weight_file}...")
        base_state_dict = load_checkpoint_state_dict(base_weight_file)
        base_missing, base_unexpected = model.load_state_dict(base_state_dict, strict=False)
        print(f"  Base Missing: {len(base_missing)}, Unexpected: {len(base_unexpected)}")
        del base_state_dict

    # Load Stage 2 fine-tuned weights
    weight_file = resolve_weight_file(stage2_dir)
    if weight_file is None:
        raise FileNotFoundError(
            f"No weights found in {stage2_dir}. "
            f"Directory contents: {describe_checkpoint_dir(stage2_dir)}"
        )

    print(f"Loading Stage 2 weights from {weight_file}...")
    state_dict = load_checkpoint_state_dict(weight_file)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    critical_prefixes = ("fusion_proj", "diffusion_head", "diff_proj", "time_embed", "image_embedder_gen")
    critical_missing = [k for k in missing if k.startswith(critical_prefixes)]
    if critical_missing:
        print(f"  [WARN] Critical missing keys: {critical_missing[:10]}")
    print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    model.eval()

    # VAE
    vae_model = WanVAE(vae_pth=args.vae_path, dtype=weight_type, device=device)

    return model, vae_model, text_tokenizer, showo_token_ids


def decode_tactile_latents(vae_model, latents):
    frames = vae_model.batch_decode(latents)
    frames = frames.squeeze(0)
    return rearrange(frames, 'c t h w -> t c h w')


def encode_video_latents(vae_model, frames, device, weight_type, deterministic=False):
    video = frames.to(device=device, dtype=weight_type)
    video = rearrange(video, 't c h w -> 1 c t h w')
    return vae_model.sample(video, deterministic=deterministic)


def resolve_latent_token_layout(model, image_latents):
    if image_latents.dim() != 5:
        raise ValueError(f"Expected 5D VAE latents, got shape={tuple(image_latents.shape)}")

    _, _, latent_frames, latent_h, latent_w = image_latents.shape
    patch_size = int(getattr(model.config, "patch_size", 2))
    latent_tokens = latent_frames * (latent_h // patch_size) * (latent_w // patch_size)
    add_time_token = bool(getattr(model.config, "add_time_embeds", False))
    segment_tokens = latent_tokens + int(add_time_token)
    max_text_len = DEFAULT_SEQ_LEN - 2 * segment_tokens - 6
    if max_text_len <= 0:
        raise ValueError(
            f"Sequence is too short for latent layout: "
            f"max_seq_len={DEFAULT_SEQ_LEN}, segment_tokens={segment_tokens}"
        )
    return latent_tokens, segment_tokens, max_text_len, add_time_token


def build_sampler_for_segment(
        segment_tokens, sampling_method, num_inference_steps, atol, rtol,
        reverse, time_shifting_factor,
):
    transport = create_transport(
        path_type="Linear",
        prediction="velocity",
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
        snr_type="lognorm",
        do_shift=True,
        seq_len=segment_tokens,
    )
    sampler = Sampler(transport)
    return sampler.sample_ode(
        sampling_method=sampling_method,
        num_steps=num_inference_steps,
        atol=atol,
        rtol=rtol,
        reverse=reverse,
        time_shifting_factor=time_shifting_factor,
    )


def prepare_gen_batch(prompts, text_tokenizer, showo_token_ids, num_visual_tokens, num_tactile_tokens, max_text_len, device):
    """Build pure generation token batches (CFG-compatible)."""
    batch_tokens, batch_tokens_null = [], []
    batch_pos, batch_pos_null = [], []

    for prompt in prompts:
        text_ids = text_tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_text_len).input_ids
        null_ids = text_tokenizer("", add_special_tokens=False, truncation=True, max_length=max_text_len).input_ids

        for ids, bucket_t, bucket_p in [(text_ids, batch_tokens, batch_pos), (null_ids, batch_tokens_null, batch_pos_null)]:
            t, _, p, _, _ = format_sequence_tactile_gen(
                text_tokens=ids, bos_id=showo_token_ids["bos_id"], eos_id=showo_token_ids["eos_id"],
                bov_id=showo_token_ids["bov_id"], eov_id=showo_token_ids["eov_id"],
                pad_id=text_tokenizer.pad_token_id, vid_pad_id=showo_token_ids["vid_pad_id"],
                num_visual_tokens=num_visual_tokens, num_tactile_tokens=num_tactile_tokens,
                max_seq_len=DEFAULT_SEQ_LEN,
            )
            bucket_t.append(t)
            bucket_p.append(p)

    return (
        torch.stack(batch_tokens, dim=0).to(device),
        torch.stack(batch_tokens_null, dim=0).to(device),
        torch.stack(batch_pos, dim=0).to(device),
        torch.stack(batch_pos_null, dim=0).to(device),
    )


def prepare_qa_batch(question, text_tokenizer, showo_token_ids, num_visual_tokens, num_tactile_tokens, max_text_len, device):
    """Build QA-formatted token batch for inference."""
    question_text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    question_ids = text_tokenizer(question_text, add_special_tokens=False, truncation=True, max_length=max_text_len // 2).input_ids
    # Empty answer placeholder for generation
    answer_ids = text_tokenizer("", add_special_tokens=False).input_ids

    tokens, _, positions, _, _ = format_sequence_tactile_qa(
        question_tokens=question_ids, answer_tokens=answer_ids,
        bos_id=showo_token_ids["bos_id"], eos_id=showo_token_ids["eos_id"],
        bov_id=showo_token_ids["bov_id"], eov_id=showo_token_ids["eov_id"],
        pad_id=text_tokenizer.pad_token_id, vid_pad_id=showo_token_ids["vid_pad_id"],
        num_visual_tokens=num_visual_tokens, num_tactile_tokens=num_tactile_tokens,
        max_seq_len=DEFAULT_SEQ_LEN,
    )
    return tokens.unsqueeze(0).to(device), positions.unsqueeze(0).to(device)


@torch.no_grad()
def generate_tactile_video(model, vae_model, text_tokenizer, showo_token_ids,
                           text_prompt, visual_video_frames, guidance_scale,
                           device, weight_type, sampling_method, num_inference_steps, atol, rtol,
                           reverse, time_shifting_factor, vae_deterministic=False):
    """Generate tactile video from text + visual condition (pure generation mode)."""
    visual_latent = encode_video_latents(
        vae_model, visual_video_frames, device, weight_type,
        deterministic=vae_deterministic,
    )
    _, segment_tokens, max_text_len, _ = resolve_latent_token_layout(model, visual_latent)

    batch_tokens, batch_tokens_null, batch_pos, batch_pos_null = prepare_gen_batch(
        [text_prompt], text_tokenizer, showo_token_ids,
        segment_tokens, segment_tokens, max_text_len, device,
    )

    z_tactile = torch.randn_like(visual_latent)
    image_latents = torch.cat([visual_latent, z_tactile], dim=0)

    if guidance_scale > 0:
        initial_latents = torch.cat([image_latents, image_latents], dim=0)
        text_tokens_cfg = torch.cat([batch_tokens, batch_tokens_null], dim=0)
        modality_positions_cfg = torch.cat([batch_pos, batch_pos_null], dim=0)
    else:
        initial_latents = image_latents
        text_tokens_cfg = batch_tokens
        modality_positions_cfg = batch_pos

    block_mask = omni_attn_mask_naive(
        text_tokens_cfg.size(0), text_tokens_cfg.size(1),
        modality_positions_cfg, device,
    ).to(weight_type)

    model_kwargs = dict(
        text_tokens=text_tokens_cfg, attention_mask=block_mask,
        modality_positions=modality_positions_cfg, output_hidden_states=True,
        max_seq_len=text_tokens_cfg.size(1), guidance_scale=guidance_scale,
        only_denoise_last_image=True,
    )

    sample_fn = build_sampler_for_segment(
        segment_tokens, sampling_method, num_inference_steps,
        atol, rtol, reverse, time_shifting_factor,
    )
    with autocast_context(device, weight_type):
        samples = sample_fn(initial_latents, model.t2i_generate, **model_kwargs)[-1]
    if guidance_scale > 0:
        samples = torch.chunk(samples, 2)[0]
    generated = samples[-1:]  # tactile latents only
    return decode_tactile_latents(vae_model, generated)


@torch.no_grad()
def build_qa_prefix_embeds(model, qa_tokens, qa_positions, image_latents, device, weight_type):
    """Reconstruct the interleaved [text + visual + tactile] embedding sequence the
    same way model.forward() does, so it can be used as a prefix for autoregressive
    answer decoding via mmu_generate.

    qa_tokens: (1, L) full QA token sequence (answer region is empty placeholder, so
               the meaningful prefix ends at the tactile EOV token).
    image_latents: (2, C, T, H, W) — [visual_latent, generated_tactile_latent].
    Returns prefix input_embeds (1, prefix_len, D) up to and including the tactile EOV.
    """
    input_embeds = model.showo.model.embed_tokens(qa_tokens)  # (1, L, D)
    dtype = input_embeds.dtype

    b, c, T, h, w = image_latents.shape
    p = model.config.patch_size
    h_, w_ = h // p, w // p

    # Dual-path image embedding (mirror of forward())
    latents_flat = rearrange(image_latents, 'b c t h w -> (b t) c h w').to(dtype)
    image_embeds_und = model.image_embedder_und(latents_flat)
    image_embeds_und = image_embeds_und.reshape(b, T, -1, model.config.clip_latent_dim)
    image_embeds_und = rearrange(image_embeds_und, 'b t l d -> (b t) l d')
    image_embeds_gen = model.image_embedder_gen(latents_flat)
    image_embeds_gen = image_embeds_gen.reshape(b, T, -1, model.config.hidden_size)
    image_embeds_gen = rearrange(image_embeds_gen, 'b t l d -> b (t l) d')

    if model.position_embedding.weight.shape[0] == model.image_position_ids.shape[-1]:
        image_embeds_und = image_embeds_und + model.position_embedding(model.image_position_ids)
    else:
        image_embeds_und = image_embeds_und + interpolate_pos_encoding(
            model.config.clip_latent_dim, model.position_embedding, h_, w_, 1,
        )
    image_embeds_und = model.und_trans(image_embeds_und)['last_hidden_state']
    image_embeds_und = image_embeds_und.reshape(b, T, image_embeds_und.shape[1], -1)
    image_embeds_und = rearrange(image_embeds_und, 'b t l d -> b (t l) d')

    image_embeds = model.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1))

    # Clean-condition time embedding (t=1.0) for both video segments
    t = torch.ones(b, device=device, dtype=weight_type)
    time_embeds = model.time_embed(t, dtype)
    if hasattr(model, 'time_embed_proj'):
        time_embeds = model.time_embed_proj(time_embeds)

    # Scatter image embeds into their modality positions (mirror of forward())
    for j, (offset, length) in enumerate(qa_positions[0]):
        offset = int(offset.item())
        length = int(length.item())
        if length == 0:
            continue
        if model.config.add_time_embeds:
            input_embeds[0, offset] = time_embeds[j]
            input_embeds[0, offset + 1:offset + length] = image_embeds[j, :length - 1]
        else:
            input_embeds[0, offset:offset + length] = image_embeds[j, :length]

    # Prefix ends right after the tactile EOV token (offset+length is tactile EOV's
    # position; +1 to include it). Everything after is empty answer placeholder/pad.
    tactile_offset = int(qa_positions[0, 1, 0].item())
    tactile_length = int(qa_positions[0, 1, 1].item())
    prefix_len = tactile_offset + tactile_length + 1  # +1 includes the EOV token
    return input_embeds[:, :prefix_len].to(weight_type)


@torch.no_grad()
def decode_qa_answer_from_latents(model, text_tokenizer, qa_tokens, qa_positions,
                                  image_latents, device, weight_type,
                                  max_new_tokens=100, top_k=None):
    with autocast_context(device, weight_type):
        prefix_embeds = build_qa_prefix_embeds(
            model, qa_tokens, qa_positions, image_latents, device, weight_type,
        )
        gen_attn_mask = omni_attn_mask_naive(
            1, prefix_embeds.size(1), qa_positions, device,
        ).to(weight_type)
        output_tokens = model.mmu_generate(
            input_embeds=prefix_embeds,
            attention_mask=gen_attn_mask,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            eos_token=text_tokenizer.eos_token_id,
        )
    answer_ids = [tok.item() if torch.is_tensor(tok) else tok for tok in output_tokens]
    return text_tokenizer.decode(answer_ids, skip_special_tokens=True)


@torch.no_grad()
def generate_qa_answer(model, vae_model, text_tokenizer, showo_token_ids,
                       question, visual_video_frames, device, weight_type,
                       sampling_method, num_inference_steps, atol, rtol,
                       reverse, time_shifting_factor, max_new_tokens=100, top_k=None):
    """QA mode: generate tactile video + answer text from question + visual condition."""
    visual_latent = encode_video_latents(vae_model, visual_video_frames, device, weight_type)
    _, segment_tokens, max_text_len, _ = resolve_latent_token_layout(model, visual_latent)

    # Build QA sequence with empty answer
    qa_tokens, qa_positions = prepare_qa_batch(
        question, text_tokenizer, showo_token_ids,
        segment_tokens, segment_tokens, max_text_len, device,
    )

    # Generate tactile video + answer
    z_tactile = torch.randn_like(visual_latent)
    image_latents_qa = torch.cat([visual_latent, z_tactile], dim=0)

    block_mask_qa = omni_attn_mask_naive(
        1, qa_tokens.size(1), qa_positions, device,
    ).to(weight_type)

    # ODE denoising → tactile latent
    model_kwargs = dict(
        text_tokens=qa_tokens, attention_mask=block_mask_qa,
        modality_positions=qa_positions, output_hidden_states=True,
        max_seq_len=qa_tokens.size(1), guidance_scale=0.0,
        only_denoise_last_image=True,
    )
    sample_fn = build_sampler_for_segment(
        segment_tokens, sampling_method, num_inference_steps,
        atol, rtol, reverse, time_shifting_factor,
    )
    with autocast_context(device, weight_type):
        samples_qa = sample_fn(image_latents_qa, model.t2i_generate, **model_kwargs)[-1]
    generated_tactile = decode_tactile_latents(vae_model, samples_qa[-1:])

    answer_text = decode_qa_answer_from_latents(
        model, text_tokenizer, qa_tokens, qa_positions, samples_qa,
        device, weight_type, max_new_tokens=max_new_tokens, top_k=top_k,
    )
    return generated_tactile, answer_text


@torch.no_grad()
def generate_qa_answer_from_tactile_condition(
        model, vae_model, text_tokenizer, showo_token_ids,
        question, visual_video_frames, tactile_video_frames,
        device, weight_type, max_new_tokens=100, top_k=None,
        vae_deterministic=False,
):
    """Decode an answer from real/generated tactile condition without denoising.

    This is the phase-2 path for two-stage evaluation: visual and tactile videos
    are both encoded as clean VAE latents, then the answer is decoded
    autoregressively through model.mmu_generate().
    """
    visual_latent = encode_video_latents(
        vae_model, visual_video_frames, device, weight_type,
        deterministic=vae_deterministic,
    )
    _, segment_tokens, max_text_len, _ = resolve_latent_token_layout(model, visual_latent)

    tactile_latent = encode_video_latents(
        vae_model, tactile_video_frames, device, weight_type,
        deterministic=vae_deterministic,
    )

    qa_tokens, qa_positions = prepare_qa_batch(
        question, text_tokenizer, showo_token_ids,
        segment_tokens, segment_tokens, max_text_len, device,
    )

    image_latents = torch.cat([visual_latent, tactile_latent], dim=0)
    return decode_qa_answer_from_latents(
        model, text_tokenizer, qa_tokens, qa_positions, image_latents,
        device, weight_type, max_new_tokens=max_new_tokens, top_k=top_k,
    )


# ==============================================================================
# Frame loading utilities (same as inference_tactile_video.py)
# ==============================================================================
def select_frame_indices(indices, num_frames, sampling_mode="contiguous", clip_start=0):
    if len(indices) == 0:
        raise ValueError("No frame indices available")
    ordered = sorted(indices)
    if len(ordered) >= num_frames:
        if sampling_mode == "contiguous":
            max_start = len(ordered) - num_frames
            start = min(max(clip_start, 0), max_start) if clip_start >= 0 else max(max_start + clip_start + 1, 0)
            return ordered[start:start + num_frames]
        sel = np.linspace(0, len(ordered) - 1, num_frames, dtype=int).tolist()
        return [ordered[p] for p in sel]
    return ordered + [ordered[-1]] * (num_frames - len(ordered))


def load_video_frames(frame_dir, num_frames, image_size=DEFAULT_RESOLUTION,
                      frame_indices=None, sampling_mode="contiguous", clip_start=0, return_indices=False):
    from torchvision import transforms
    frame_map = {int(os.path.splitext(f)[0]): f for f in os.listdir(frame_dir)
                 if f.endswith(".png") and os.path.splitext(f)[0].isdigit()}
    if not frame_map:
        raise ValueError(f"No PNG files in {frame_dir}")
    available = sorted(frame_map)
    src = [i for i in (frame_indices or available) if i in frame_map] or available
    selected = select_frame_indices(src, num_frames, sampling_mode, clip_start)

    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    frames = torch.stack([transform(Image.open(os.path.join(frame_dir, frame_map[i])).convert('RGB')) for i in selected])
    return (frames, selected) if return_indices else frames


def save_video_frames(frames_tensor, output_path, fps=2):
    output_dir = os.path.dirname(output_path)
    ext = os.path.splitext(output_path)[1].lower()
    if ext in {".mp4", ".gif"}:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        try:
            import imageio
            frames = (frames_tensor.float().permute(0, 2, 3, 1).cpu().numpy() + 1.0) / 2.0
            frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
            imageio.mimsave(output_path, frames, fps=fps)
            print(f"Saved video to {output_path}")
            return
        except ImportError:
            output_path = os.path.splitext(output_path)[0]
    os.makedirs(output_path, exist_ok=True)
    frames_np = denorm_vid(frames_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4))[0]
    for i, frame in enumerate(frames_np):
        Image.fromarray(frame).save(os.path.join(output_path, f"frame_{i:04d}.png"))


def save_condition_videos(args, output_path, visual_frames, tactile_frames=None):
    if not args.save_conditions:
        return
    stem = os.path.splitext(output_path)[0]
    save_video_frames(visual_frames, f"{stem}_visual.mp4", fps=args.fps)
    if tactile_frames is not None:
        save_video_frames(tactile_frames, f"{stem}_target.mp4", fps=args.fps)


def write_jsonl(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ==============================================================================
# Main entry points
# ==============================================================================
def run_single_inference(args, model, vae_model, text_tokenizer, showo_token_ids, device, weight_type):
    print(f"Loading visual frames from {args.visual_video_dir}...")
    visual_frames = load_video_frames(args.visual_video_dir, args.num_frames, args.image_size,
                                      sampling_mode=args.sampling_mode, clip_start=args.clip_start)
    print(f"  Loaded {visual_frames.shape[0]} frames")

    if args.qa_mode:
        print(f"QA mode: generating tactile video + answer for question...")
        print(f"  Question: {args.text[:200]}...")
        generated, answer = generate_qa_answer(
            model, vae_model, text_tokenizer, showo_token_ids,
            question=args.text, visual_video_frames=visual_frames,
            device=device, weight_type=weight_type,
            sampling_method=args.sampling_method,
            num_inference_steps=args.num_inference_steps, atol=args.atol, rtol=args.rtol,
            reverse=args.reverse, time_shifting_factor=args.time_shifting_factor,
        )
        print(f"  Answer: {answer}")
        # Save answer text alongside video
        base = os.path.splitext(args.output_path)[0]
        with open(f"{base}_answer.txt", "w") as f:
            f.write(f"Question: {args.text}\nAnswer: {answer}\n")
    else:
        print(f"Pure generation: text + visual → tactile")
        print(f"  Text: {args.text[:200]}...")
        generated = generate_tactile_video(
            model, vae_model, text_tokenizer, showo_token_ids,
            text_prompt=args.text, visual_video_frames=visual_frames,
            guidance_scale=args.guidance_scale, device=device, weight_type=weight_type,
            sampling_method=args.sampling_method,
            num_inference_steps=args.num_inference_steps, atol=args.atol, rtol=args.rtol,
            reverse=args.reverse, time_shifting_factor=args.time_shifting_factor,
        )

    save_video_frames(generated, args.output_path, fps=args.fps)
    print("Done!")


def _resolve_token_counts(model, args):
    _, segment_tokens, max_text_len, _ = resolve_video_token_layout(model, args.num_frames)
    return segment_tokens, segment_tokens, max_text_len


def build_tactile_visual_dataset(args, text_tokenizer, showo_token_ids, num_visual_tokens, num_tactile_tokens):
    return TactileVisualDataset(
        data_root=args.tactile_data_root, csv_path=args.tactile_csv_path,
        text_tokenizer=text_tokenizer, max_seq_len=DEFAULT_SEQ_LEN,
        image_size=args.image_size, latent_height=DEFAULT_LATENT_H,
        latent_width=DEFAULT_LATENT_W, num_frames=args.num_frames,
        num_visual_tokens_per_frame=DEFAULT_TOKENS_PER_FRAME,
        num_tactile_tokens_per_frame=DEFAULT_TOKENS_PER_FRAME,
        num_visual_tokens=num_visual_tokens, num_tactile_tokens=num_tactile_tokens,
        cond_dropout_prob=0.0, split=args.eval_split,
        showo_token_ids=showo_token_ids,
    )


def build_tactile_qa_dataset(args, text_tokenizer, showo_token_ids, num_visual_tokens, num_tactile_tokens):
    return TactileQADataset(
        data_root=args.tactile_data_root, csv_path=args.tactile_csv_path,
        tactile_qa_csv_path=args.tactile_qa_csv_path,
        text_tokenizer=text_tokenizer, max_seq_len=DEFAULT_SEQ_LEN,
        image_size=args.image_size, latent_height=DEFAULT_LATENT_H,
        latent_width=DEFAULT_LATENT_W, num_frames=args.num_frames,
        num_visual_tokens_per_frame=DEFAULT_TOKENS_PER_FRAME,
        num_tactile_tokens_per_frame=DEFAULT_TOKENS_PER_FRAME,
        num_visual_tokens=num_visual_tokens, num_tactile_tokens=num_tactile_tokens,
        cond_dropout_prob=0.0, split=args.eval_split,
        showo_token_ids=showo_token_ids,
    )


def run_batch_test(args, model, vae_model, text_tokenizer, showo_token_ids, device, weight_type):
    num_visual_tokens, num_tactile_tokens, _ = _resolve_token_counts(model, args)

    if args.qa_mode:
        print(f"Building QA test dataset from {args.tactile_qa_csv_path}...")
        dataset = build_tactile_qa_dataset(
            args, text_tokenizer, showo_token_ids, num_visual_tokens, num_tactile_tokens,
        )
        mode_label = "QA"
    else:
        print(f"Building test dataset from {args.tactile_csv_path}...")
        dataset = build_tactile_visual_dataset(
            args, text_tokenizer, showo_token_ids, num_visual_tokens, num_tactile_tokens,
        )
        mode_label = "Pure Gen"

    print(f"  {mode_label} mode: {len(dataset)} samples in '{args.eval_split}' split")

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = []
    success = 0

    for idx, sample in enumerate(dataset.samples, start=1):
        object_name = sample.get('object_name', f'sample_{idx - 1}')
        print(f"\n[{idx}/{len(dataset.samples)}] {object_name}")

        try:
            selected_indices = dataset._select_frame_indices(sample["frame_indices"])
            visual_frames, sel_indices = dataset.load_sample_video(
                sample, "visual", selected_indices
            )
            tactile_target = None
            if args.save_conditions:
                tactile_target, _ = dataset.load_sample_video(
                    sample, "tactile", selected_indices
                )

            if args.qa_mode:
                question = sample.get('question', '')
                generated, answer = generate_qa_answer(
                    model, vae_model, text_tokenizer, showo_token_ids,
                    question=question, visual_video_frames=visual_frames,
                    device=device, weight_type=weight_type,
                    sampling_method=args.sampling_method,
                    num_inference_steps=args.num_inference_steps,
                    atol=args.atol, rtol=args.rtol,
                    reverse=args.reverse,
                    time_shifting_factor=args.time_shifting_factor,
                )
                gt_answer = sample.get('answer', '')
                entry = {
                    'object_name': object_name,
                    'question': question,
                    'predicted_answer': answer,
                    'gt_answer': gt_answer,
                    'qa_type': sample.get('qa_type', ''),
                    'frame_indices': sel_indices,
                }
            else:
                text = sample.get('text', '')
                generated = generate_tactile_video(
                    model, vae_model, text_tokenizer, showo_token_ids,
                    text_prompt=text, visual_video_frames=visual_frames,
                    guidance_scale=args.guidance_scale,
                    device=device, weight_type=weight_type,
                    sampling_method=args.sampling_method,
                    num_inference_steps=args.num_inference_steps,
                    atol=args.atol, rtol=args.rtol,
                    reverse=args.reverse,
                    time_shifting_factor=args.time_shifting_factor,
                )
                entry = {
                    'object_name': object_name,
                    'text': text,
                    'frame_indices': sel_indices,
                }

            out_path = os.path.join(args.output_dir, f"{object_name}.mp4")
            save_video_frames(generated, out_path, fps=args.fps)
            save_condition_videos(args, out_path, visual_frames, tactile_target)
            entry['output_path'] = out_path
            entry['status'] = 'success'
            manifest.append(entry)
            success += 1
            print(f"  ✓ {object_name}")

        except Exception as exc:
            print(f"  ✗ {object_name}: {exc}")
            manifest.append({'object_name': object_name, 'status': 'failed', 'error': str(exc)})

    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
    write_jsonl(manifest_path, manifest)
    print(f"\nDone: {success}/{len(dataset)} successful. Manifest saved to {manifest_path}")


def run_two_stage_eval(args, model, vae_model, text_tokenizer, showo_token_ids, device, weight_type):
    num_visual_tokens, num_tactile_tokens, _ = _resolve_token_counts(model, args)

    if args.qa_condition_source != "target_tactile":
        raise ValueError(
            f"Unsupported --qa_condition_source={args.qa_condition_source}. "
            "This implementation currently supports only 'target_tactile'."
        )

    print(f"[Phase 2/2] Building QA dataset from {args.tactile_qa_csv_path}...")
    qa_dataset = build_tactile_qa_dataset(
        args, text_tokenizer, showo_token_ids, num_visual_tokens, num_tactile_tokens,
    )
    qa_count = min(args.qa_sample_size, len(qa_dataset.samples))
    sampled_indices = random.Random(args.qa_sample_seed).sample(
        range(len(qa_dataset.samples)), qa_count
    ) if qa_count > 0 else []
    print(
        f"  QA samples: {len(qa_dataset)} total, "
        f"sampling {qa_count} with seed {args.qa_sample_seed}"
    )

    qa_results_path = args.qa_results_path or os.path.join(args.output_dir, "qa_results.jsonl")
    os.makedirs(os.path.dirname(qa_results_path) or ".", exist_ok=True)

    qa_success = 0
    qa_failures = 0
    with open(qa_results_path, "w", encoding="utf-8") as f:
        for ordinal, sample_index in enumerate(sampled_indices, start=1):
            sample = qa_dataset.samples[sample_index]
            object_name = sample.get("object_name", f"sample_{sample_index}")
            question = sample.get("question", "")
            print(f"\n[Phase 2][{ordinal}/{qa_count}] {object_name} | sample_index={sample_index}")

            entry = {
                "sample_index": sample_index,
                "object_name": object_name,
                "question": question,
                "gt_answer": sample.get("answer", ""),
                "qa_type": sample.get("qa_type", ""),
                "qa_condition_source": args.qa_condition_source,
            }

            try:
                selected_indices = qa_dataset._select_frame_indices(sample["frame_indices"])
                visual_frames, sel_indices = qa_dataset.load_sample_video(
                    sample, "visual", selected_indices
                )
                tactile_frames, _ = qa_dataset.load_sample_video(
                    sample, "tactile", selected_indices
                )
                answer = generate_qa_answer_from_tactile_condition(
                    model, vae_model, text_tokenizer, showo_token_ids,
                    question=question,
                    visual_video_frames=visual_frames,
                    tactile_video_frames=tactile_frames,
                    device=device,
                    weight_type=weight_type,
                    vae_deterministic=args.vae_deterministic,
                )
                entry.update({
                    "predicted_answer": answer,
                    "frame_indices": sel_indices,
                    "status": "success",
                })
                qa_success += 1
                print(f"  Answer: {answer}")
            except Exception as exc:
                entry.update({
                    "predicted_answer": "",
                    "frame_indices": [],
                    "status": "failed",
                    "error": str(exc),
                })
                qa_failures += 1
                print(f"  ✗ QA failed for {object_name}: {exc}")

            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[Phase 2] Done: {qa_success}/{qa_count} successful. Results: {qa_results_path}")

    generated_dir = os.path.join(args.output_dir, "generated")
    os.makedirs(generated_dir, exist_ok=True)

    print(f"\n[Phase 1/2] Building tactile generation dataset from {args.tactile_csv_path}...")
    gen_dataset = build_tactile_visual_dataset(
        args, text_tokenizer, showo_token_ids, num_visual_tokens, num_tactile_tokens,
    )
    print(f"  Generation samples: {len(gen_dataset)} in '{args.eval_split}' split")

    gen_manifest = []
    gen_success = 0
    for idx, sample in enumerate(gen_dataset.samples, start=1):
        object_name = sample.get("object_name", f"sample_{idx - 1}")
        print(f"\n[Phase 1][{idx}/{len(gen_dataset.samples)}] {object_name}")

        try:
            selected_indices = gen_dataset._select_frame_indices(sample["frame_indices"])
            visual_frames, sel_indices = gen_dataset.load_sample_video(
                sample, "visual", selected_indices
            )
            tactile_target = None
            if args.save_conditions:
                tactile_target, _ = gen_dataset.load_sample_video(
                    sample, "tactile", selected_indices
                )

            generated = generate_tactile_video(
                model, vae_model, text_tokenizer, showo_token_ids,
                text_prompt=sample.get("text", ""),
                visual_video_frames=visual_frames,
                guidance_scale=args.guidance_scale,
                device=device,
                weight_type=weight_type,
                sampling_method=args.sampling_method,
                num_inference_steps=args.num_inference_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse,
                time_shifting_factor=args.time_shifting_factor,
                vae_deterministic=args.vae_deterministic,
            )

            out_path = os.path.join(generated_dir, f"{object_name}.mp4")
            save_video_frames(generated, out_path, fps=args.fps)
            save_condition_videos(args, out_path, visual_frames, tactile_target)

            gen_manifest.append({
                "object_name": object_name,
                "text": sample.get("text", ""),
                "frame_indices": sel_indices,
                "output_path": out_path,
                "status": "success",
            })
            gen_success += 1
            print(f"  ✓ generated {object_name}")
        except Exception as exc:
            print(f"  ✗ generation failed for {object_name}: {exc}")
            gen_manifest.append({
                "object_name": object_name,
                "status": "failed",
                "error": str(exc),
            })

    gen_manifest_path = os.path.join(args.output_dir, "tactile_generation_manifest.jsonl")
    write_jsonl(gen_manifest_path, gen_manifest)
    print(f"[Phase 1] Done: {gen_success}/{len(gen_dataset)} successful. Manifest: {gen_manifest_path}")

    qa_summary = {
        "total_attempted": qa_count,
        "success_count": qa_success,
        "failure_count": qa_failures,
        "qa_sample_seed": args.qa_sample_seed,
        "qa_sample_size": args.qa_sample_size,
        "qa_condition_source": args.qa_condition_source,
        "qa_results_path": qa_results_path,
        "generation_total": len(gen_dataset),
        "generation_success_count": gen_success,
        "generation_manifest_path": gen_manifest_path,
    }
    qa_summary_path = os.path.splitext(qa_results_path)[0] + "_summary.json"
    with open(qa_summary_path, "w", encoding="utf-8") as f:
        json.dump(qa_summary, f, ensure_ascii=False, indent=2)
    print(f"[Two-stage Eval] Summary: {qa_summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Tactile Video Inference")
    # Model paths
    parser.add_argument("--stage2_checkpoint", type=str, required=True,
                        help="Path to Stage 2 checkpoint (checkpoint-N/unwrapped_model)")
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--llm_path", type=str, required=True)
    parser.add_argument("--showo_path", type=str, default=None,
                        help="Base Show-o2 model (optional; if omitted, creates from llm_path)")
    parser.add_argument("--siglip_path", type=str, default=None,
                        help="SigLIP model path (optional for Stage 2)")
    # Generation params
    parser.add_argument("--text", type=str, default="Contact: True, Material: rubber.",
                        help="Text prompt (or question in QA mode)")
    parser.add_argument("--visual_video_dir", type=str, default=None,
                        help="Directory of visual video frames (single inference)")
    parser.add_argument("--output_path", type=str, default="output_tactile_stage2.mp4")
    parser.add_argument("--output_dir", type=str, default="outputs/stage2_inference")
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--image_size", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--sampling_mode", type=str, default="contiguous")
    parser.add_argument("--clip_start", type=int, default=0)
    # ODE params
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--sampling_method", type=str, default="euler")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--time_shifting_factor", type=float, default=3.0)
    parser.add_argument("--vae_deterministic", action="store_true")
    # Mode
    parser.add_argument("--batch_test", action="store_true", help="Run batch inference on entire test split")
    parser.add_argument("--qa_mode", action="store_true", help="Enable QA mode (question → answer + tactile video)")
    parser.add_argument("--two_stage_eval", action="store_true",
                        help="Run phase 1 tactile generation for all objects, then phase 2 QA on sampled questions")
    parser.add_argument("--qa_sample_size", type=int, default=50,
                        help="Number of QA rows sampled for --two_stage_eval")
    parser.add_argument("--qa_sample_seed", type=int, default=42,
                        help="Random seed for reproducible QA sampling in --two_stage_eval")
    parser.add_argument("--qa_condition_source", type=str, default="target_tactile",
                        choices=["target_tactile"],
                        help="Tactile condition source for phase-2 QA")
    parser.add_argument("--qa_results_path", type=str, default=None,
                        help="JSONL path for phase-2 QA results; defaults to output_dir/qa_results.jsonl")
    # Data paths (batch test)
    parser.add_argument("--tactile_data_root", type=str, default=None)
    parser.add_argument("--tactile_csv_path", type=str, default=None)
    parser.add_argument("--tactile_qa_csv_path", type=str, default="tac_QA/tactile_qa_pairs.csv")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--save_conditions", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_type = torch.bfloat16

    print("=" * 60)
    print("Stage 2 Tactile Video Inference")
    print(f"  Checkpoint: {args.stage2_checkpoint}")
    if args.two_stage_eval:
        mode_name = "Two-Stage Eval"
    else:
        mode_name = "QA" if args.qa_mode else "Pure Generation"
    print(f"  Mode: {mode_name}")
    print(f"  Batch: {args.batch_test}")
    print("=" * 60)

    print("[1/4] Loading model...")
    model, vae_model, text_tokenizer, showo_token_ids = load_model(args, device, weight_type)
    print("[2/4] Model loaded.")
    print("[3/4] Running inference...")

    if args.two_stage_eval:
        run_two_stage_eval(args, model, vae_model, text_tokenizer, showo_token_ids, device, weight_type)
    elif args.batch_test:
        run_batch_test(args, model, vae_model, text_tokenizer, showo_token_ids, device, weight_type)
    else:
        if args.visual_video_dir is None:
            parser.error("--visual_video_dir is required for single inference")
        run_single_inference(args, model, vae_model, text_tokenizer, showo_token_ids, device, weight_type)


if __name__ == "__main__":
    main()
