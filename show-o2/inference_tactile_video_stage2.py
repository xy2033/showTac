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
import json
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from datasets import TactileVisualDataset
from datasets.tactile_qa_dataset import TactileQADataset
from datasets.utils import format_sequence_tactile_gen, format_sequence_tactile_qa
from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
from models.misc import get_text_tokenizer
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


def load_checkpoint_state_dict(weight_file):
    if weight_file.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(weight_file, device="cpu")
    return torch.load(weight_file, map_location="cpu")


def load_model(args, device, weight_type):
    """Load the full Show-o2 model with Stage 2 fine-tuned weights."""
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        args.llm_path, add_showo_tokens=True, return_showo_token_ids=True,
        llm_name=path_to_llm_name[args.llm_path],
    )

    # Load base Show-o2 architecture
    if args.showo_path and os.path.isdir(args.showo_path):
        print(f"Loading base Show-o2 from {args.showo_path}...")
        model = Showo2Qwen2_5.from_pretrained(args.showo_path, use_safetensors=False).to(device)
    else:
        cfg = dict(
            model_name="Showo2", llm_model_path=args.llm_path,
            hidden_size=1536, image_latent_dim=16,
            image_latent_height=DEFAULT_LATENT_H, image_latent_width=DEFAULT_LATENT_W,
            patch_size=2, num_diffusion_layers=10, clip_latent_dim=1152,
            add_qk_norm=True, add_time_embeds=True,
        )
        model = Showo2Qwen2_5(**cfg).to(device)

    # Load Stage 2 fine-tuned weights
    stage2_dir = resolve_checkpoint_dir(args.stage2_checkpoint)
    weight_file = None
    for fname in ("pytorch_model.bin", "model.safetensors"):
        candidate = os.path.join(stage2_dir, fname)
        if os.path.exists(candidate):
            weight_file = candidate
            break
    if weight_file is None:
        raise FileNotFoundError(f"No weights found in {stage2_dir}")

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

    # Flow matching sampler
    from omegaconf import OmegaConf
    transport_config = OmegaConf.create(dict(
        path_type="Linear", prediction="velocity", loss_weight=None,
        train_eps=None, sample_eps=None, snr_type="lognorm",
        sampling_method=args.sampling_method,
        num_inference_steps=args.num_inference_steps,
        atol=args.atol, rtol=args.rtol, reverse=args.reverse,
        do_shift=True, time_shifting_factor=args.time_shifting_factor,
    ))
    transport = create_transport(
        path_type=transport_config.path_type,
        prediction=transport_config.prediction,
        loss_weight=transport_config.loss_weight,
        train_eps=transport_config.train_eps,
        sample_eps=transport_config.sample_eps,
        snr_type=transport_config.snr_type,
        do_shift=transport_config.do_shift,
        seq_len=DEFAULT_TOKENS_PER_FRAME * args.num_frames,
    )
    sampler = Sampler(transport)

    return model, vae_model, text_tokenizer, showo_token_ids, sampler


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

    tokens, labels, positions, text_mask, image_mask = format_sequence_tactile_qa(
        question_tokens=question_ids, answer_tokens=answer_ids,
        bos_id=showo_token_ids["bos_id"], eos_id=showo_token_ids["eos_id"],
        bov_id=showo_token_ids["bov_id"], eov_id=showo_token_ids["eov_id"],
        pad_id=text_tokenizer.pad_token_id, vid_pad_id=showo_token_ids["vid_pad_id"],
        num_visual_tokens=num_visual_tokens, num_tactile_tokens=num_tactile_tokens,
        max_seq_len=DEFAULT_SEQ_LEN,
    )
    return tokens.unsqueeze(0).to(device), positions.unsqueeze(0).to(device), text_mask.unsqueeze(0).to(device), image_mask.unsqueeze(0).to(device)


@torch.no_grad()
def generate_tactile_video(model, vae_model, text_tokenizer, showo_token_ids, sampler,
                           text_prompt, visual_video_frames, num_frames, guidance_scale,
                           device, weight_type, num_visual_tokens, num_tactile_tokens,
                           max_text_len, sampling_method, num_inference_steps, atol, rtol,
                           reverse, time_shifting_factor, vae_deterministic=False):
    """Generate tactile video from text + visual condition (pure generation mode)."""
    # Encode visual video
    visual_input = visual_video_frames.unsqueeze(0).to(device=device, dtype=weight_type)
    visual_latent = vae_model.sample(visual_input)

    batch_tokens, batch_tokens_null, batch_pos, batch_pos_null = prepare_gen_batch(
        [text_prompt], text_tokenizer, showo_token_ids,
        num_visual_tokens, num_tactile_tokens, max_text_len, device,
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

    sample_fn = sampler.sample_ode(
        sampling_method=sampling_method, num_steps=num_inference_steps,
        atol=atol, rtol=rtol, reverse=reverse,
        time_shifting_factor=time_shifting_factor,
    )
    samples = sample_fn(initial_latents, model.t2i_generate, **model_kwargs)[-1]
    if guidance_scale > 0:
        samples = torch.chunk(samples, 2)[0]
    generated = samples[-1:]  # tactile latents only
    return vae_model.batch_decode(generated)


@torch.no_grad()
def generate_qa_answer(model, vae_model, text_tokenizer, showo_token_ids, sampler,
                       question, visual_video_frames, num_frames, guidance_scale,
                       device, weight_type, num_visual_tokens, num_tactile_tokens,
                       max_text_len, sampling_method, num_inference_steps, atol, rtol,
                       reverse, time_shifting_factor):
    """QA mode: generate tactile video + answer text from question + visual condition."""
    # Encode visual video
    visual_input = visual_video_frames.unsqueeze(0).to(device=device, dtype=weight_type)
    visual_latent = vae_model.sample(visual_input)

    # Build QA sequence with empty answer
    qa_tokens, qa_positions, qa_text_mask, qa_image_mask = prepare_qa_batch(
        question, text_tokenizer, showo_token_ids,
        num_visual_tokens, num_tactile_tokens, max_text_len, device,
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
    sample_fn = sampler.sample_ode(
        sampling_method=sampling_method, num_steps=num_inference_steps,
        atol=atol, rtol=rtol, reverse=reverse,
        time_shifting_factor=time_shifting_factor,
    )
    samples_qa = sample_fn(image_latents_qa, model.t2i_generate, **model_kwargs)[-1]
    generated_tactile = vae_model.batch_decode(samples_qa[-1:])

    # Decode answer text from NTP logits (forward pass on completed tactile)
    logits, _, _ = model(
        text_tokens=qa_tokens, image_latents=samples_qa,
        t=torch.ones(samples_qa.shape[0], device=device).to(weight_type),
        attention_mask=block_mask_qa, text_masks=qa_text_mask,
        image_masks=qa_image_mask,
        text_labels=torch.full_like(qa_tokens, -100),
        image_labels=torch.zeros_like(samples_qa[:, :1, :, :, :]),
        modality_positions=qa_positions, output_hidden_states=True,
        max_seq_len=qa_tokens.size(1), device=device,
    )

    # Extract predicted answer tokens
    pred_ids = logits[0, :-1].argmax(dim=-1)
    # Find answer region (non -100 positions in the sequence that come after video)
    # The answer region starts after the tactile video EOV token
    tactile_eov_pos = qa_positions[0, 1, 0] + qa_positions[0, 1, 1]  # tactile_offset + tactile_length
    answer_start = tactile_eov_pos + 1  # +1 for EOV
    # Use token after the EOV as start; decode until EOS or pad
    answer_tokens = []
    for pos in range(answer_start, min(answer_start + 100, len(pred_ids))):
        tid = pred_ids[pos].item()
        if tid == text_tokenizer.eos_token_id or tid == text_tokenizer.pad_token_id:
            break
        answer_tokens.append(tid)

    answer_text = text_tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return generated_tactile, answer_text


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


# ==============================================================================
# Main entry points
# ==============================================================================
def run_single_inference(args, model, vae_model, text_tokenizer, showo_token_ids, sampler, device, weight_type):
    num_visual_tokens, num_tactile_tokens, max_text_len = _resolve_token_counts(model, args)

    print(f"Loading visual frames from {args.visual_video_dir}...")
    visual_frames = load_video_frames(args.visual_video_dir, args.num_frames, args.image_size,
                                      sampling_mode=args.sampling_mode, clip_start=args.clip_start)
    print(f"  Loaded {visual_frames.shape[0]} frames")

    if args.qa_mode:
        print(f"QA mode: generating tactile video + answer for question...")
        print(f"  Question: {args.text[:200]}...")
        generated, answer = generate_qa_answer(
            model, vae_model, text_tokenizer, showo_token_ids, sampler,
            question=args.text, visual_video_frames=visual_frames, num_frames=args.num_frames,
            guidance_scale=args.guidance_scale, device=device, weight_type=weight_type,
            num_visual_tokens=num_visual_tokens, num_tactile_tokens=num_tactile_tokens,
            max_text_len=max_text_len, sampling_method=args.sampling_method,
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
            model, vae_model, text_tokenizer, showo_token_ids, sampler,
            text_prompt=args.text, visual_video_frames=visual_frames, num_frames=args.num_frames,
            guidance_scale=args.guidance_scale, device=device, weight_type=weight_type,
            num_visual_tokens=num_visual_tokens, num_tactile_tokens=num_tactile_tokens,
            max_text_len=max_text_len, sampling_method=args.sampling_method,
            num_inference_steps=args.num_inference_steps, atol=args.atol, rtol=args.rtol,
            reverse=args.reverse, time_shifting_factor=args.time_shifting_factor,
        )

    save_video_frames(generated, args.output_path, fps=args.fps)
    print("Done!")


def _resolve_token_counts(model, args):
    _, segment_tokens, max_text_len, _ = resolve_video_token_layout(model, args.num_frames)
    return segment_tokens, segment_tokens, max_text_len


def run_batch_test(args, model, vae_model, text_tokenizer, showo_token_ids, sampler, device, weight_type):
    num_visual_tokens, num_tactile_tokens, max_text_len = _resolve_token_counts(model, args)

    if args.qa_mode:
        print(f"Building QA test dataset from {args.tactile_qa_csv_path}...")
        dataset = TactileQADataset(
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
        mode_label = "QA"
    else:
        print(f"Building test dataset from {args.tactile_csv_path}...")
        dataset = TactileVisualDataset(
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
        mode_label = "Pure Gen"

    print(f"  {mode_label} mode: {len(dataset)} samples in '{args.eval_split}' split")

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = []
    success = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample is None:
            continue

        object_name = sample.get('object_names', f'sample_{idx}')
        print(f"\n[{idx + 1}/{len(dataset)}] {object_name}")

        try:
            # Load visual frames
            visual_frames, sel_indices = load_video_frames(
                sample['visual_dir'], args.num_frames, args.image_size,
                frame_indices=sample.get('frame_indices'),
                sampling_mode=args.sampling_mode, return_indices=True,
            )

            if args.qa_mode:
                question = sample.get('question', sample.get('texts', ''))
                generated, answer = generate_qa_answer(
                    model, vae_model, text_tokenizer, showo_token_ids, sampler,
                    question=question, visual_video_frames=visual_frames,
                    num_frames=args.num_frames, guidance_scale=args.guidance_scale,
                    device=device, weight_type=weight_type,
                    num_visual_tokens=num_visual_tokens,
                    num_tactile_tokens=num_tactile_tokens,
                    max_text_len=max_text_len,
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
                text = sample.get('texts', '')
                generated = generate_tactile_video(
                    model, vae_model, text_tokenizer, showo_token_ids, sampler,
                    text_prompt=text, visual_video_frames=visual_frames,
                    num_frames=args.num_frames, guidance_scale=args.guidance_scale,
                    device=device, weight_type=weight_type,
                    num_visual_tokens=num_visual_tokens,
                    num_tactile_tokens=num_tactile_tokens,
                    max_text_len=max_text_len,
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
            entry['output_path'] = out_path
            entry['status'] = 'success'
            manifest.append(entry)
            success += 1
            print(f"  ✓ {object_name}")

        except Exception as exc:
            print(f"  ✗ {object_name}: {exc}")
            manifest.append({'object_name': object_name, 'status': 'failed', 'error': str(exc)})

    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
    with open(manifest_path, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nDone: {success}/{len(dataset)} successful. Manifest saved to {manifest_path}")


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
    print(f"  Mode: {'QA' if args.qa_mode else 'Pure Generation'}")
    print(f"  Batch: {args.batch_test}")
    print("=" * 60)

    print("[1/4] Loading model...")
    model, vae_model, text_tokenizer, showo_token_ids, sampler = load_model(args, device, weight_type)
    print("[2/4] Model loaded.")
    print("[3/4] Running inference...")

    if args.batch_test:
        run_batch_test(args, model, vae_model, text_tokenizer, showo_token_ids, sampler, device, weight_type)
    else:
        if args.visual_video_dir is None:
            parser.error("--visual_video_dir is required for single inference")
        run_single_inference(args, model, vae_model, text_tokenizer, showo_token_ids, sampler, device, weight_type)


if __name__ == "__main__":
    main()
