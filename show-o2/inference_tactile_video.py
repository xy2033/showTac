# coding=utf-8
# Copyright 2025 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tactile-Video Generation Inference (Stage 1 Checkpoint).

Takes a physical text description + visual video (img_gelsight frames) and
generates a tactile video via flow-matching ODE denoising.

可以直接使用阶段一的训练结果进行推理，不需要阶段二。
阶段一已训练了 fusion_proj + diffusion_head + time_embed，
这三个组件是 Flow-Matching 去噪生成触觉 latent 的核心。

Usage (集群离线模式):
    python inference_tactile_video.py \
        --stage1_checkpoint outputs/showo2-1.5b-tactile-stage-1/checkpoint-final/unwrapped_model \
        --vae_path /path/to/Wan2.1_VAE.pth \
        --llm_path /path/to/Qwen2.5-1.5B-Instruct \
        --showo_path /path/to/show-o2-1.5B \
        --siglip_path /path/to/siglip-so400m-patch14-384 \
        --text "Contact: True, Material: Rough Wood" \
        --visual_video_dir /path/to/img_gelsight/ \
        --output_path output_tactile.mp4 \
        --num_frames 5
"""

import argparse
import json
import os
from contextlib import nullcontext
import numpy as np
from PIL import Image

import torch
from einops import rearrange

from datasets import TactileVisualDataset
from datasets.utils import format_sequence_tactile_gen
from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
from models.misc import get_text_tokenizer
from utils import path_to_llm_name, denorm_vid
from transport import Sampler, create_transport


# ==============================================================================
# 默认配置 (与训练时一致，可通过命令行覆盖)
# ==============================================================================
DEFAULT_RESOLUTION = 432           # 帧 resize 分辨率
DEFAULT_NUM_FRAMES = 5             # 采样帧数
DEFAULT_LATENT_H = 27              # VAE latent 高度: 432/8/2 = 27
DEFAULT_LATENT_W = 27              # VAE latent 宽度: 432/8/2 = 27
DEFAULT_TOKENS_PER_FRAME = 729     # 每帧 token 数: 27*27 = 729
DEFAULT_SEQ_LEN = 8192             # 最大序列长度


def autocast_context(device, dtype):
    device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    if device_type == "cuda" and dtype in (torch.bfloat16, torch.float16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def resolve_video_token_layout(model, num_frames, time_embed_layout):
    patch_size = int(getattr(model.config, "patch_size", 2))
    latent_h = int(getattr(model.config, "image_latent_height", DEFAULT_LATENT_H))
    latent_w = int(getattr(model.config, "image_latent_width", DEFAULT_LATENT_W))
    latent_tokens = num_frames * latent_h * latent_w
    model_adds_time = bool(getattr(model.config, "add_time_embeds", False))
    add_time_token = model_adds_time

    if time_embed_layout == "with_time_token":
        add_time_token = True
    elif time_embed_layout == "without_time_token":
        if model_adds_time:
            raise ValueError(
                "This checkpoint has add_time_embeds=True, so inference must reserve "
                "one time-token position per video segment. Use --time_embed_layout auto "
                "or --time_embed_layout with_time_token."
            )
        add_time_token = False

    segment_tokens = latent_tokens + int(add_time_token)
    max_text_len = DEFAULT_SEQ_LEN - 2 * segment_tokens - 6
    if max_text_len <= 0:
        raise ValueError(
            f"Sequence is too short for {num_frames} frames: "
            f"max_seq_len={DEFAULT_SEQ_LEN}, segment_tokens={segment_tokens}"
        )

    return latent_tokens, segment_tokens, max_text_len, add_time_token


def resolve_checkpoint_dir(checkpoint_path):
    """Accept either an unwrapped_model directory or its parent checkpoint directory."""
    if os.path.exists(os.path.join(checkpoint_path, "config.json")):
        return checkpoint_path
    unwrapped = os.path.join(checkpoint_path, "unwrapped_model")
    if os.path.exists(os.path.join(unwrapped, "config.json")):
        return unwrapped
    return checkpoint_path


def resolve_weight_file(model_path):
    if os.path.isdir(model_path):
        # Prefer single-file checkpoints, then fall back to sharded index files
        # (save_pretrained shards once the state dict exceeds max_shard_size).
        for filename in (
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.bin.index.json",
            "model.safetensors.index.json",
            "diffusion_pytorch_model.bin.index.json",
            "diffusion_pytorch_model.safetensors.index.json",
        ):
            weight_file = os.path.join(model_path, filename)
            if os.path.exists(weight_file):
                return weight_file
    elif os.path.exists(model_path):
        return model_path
    return None


def load_checkpoint_state_dict(weight_file):
    # Sharded checkpoint: merge every shard listed in the index's weight_map.
    if weight_file.endswith(".index.json"):
        shard_dir = os.path.dirname(weight_file)
        with open(weight_file, "r", encoding="utf-8") as index_file:
            index = json.load(index_file)
        merged = {}
        for shard_name in sorted(set(index["weight_map"].values())):
            merged.update(load_checkpoint_state_dict(os.path.join(shard_dir, shard_name)))
        return merged
    if weight_file.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is required to load model.safetensors checkpoints"
            ) from exc
        return load_file(weight_file, device="cpu")
    return torch.load(weight_file, map_location="cpu")


def check_checkpoint_load(model, missing, unexpected, allow_partial=False, base_loaded=False):
    critical_prefixes = (
        "fusion_proj",
        "diffusion_head",
        "diff_proj",
        "time_embed",
        "time_embed_proj",
        "image_embedder_gen",
    )
    critical_missing = [
        key for key in missing
        if key.startswith(critical_prefixes)
    ]
    if critical_missing:
        print(f"  Critical missing keys: {critical_missing[:20]}")
    if missing:
        print(f"  Missing keys count: {len(missing)}; first keys: {missing[:20]}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:20]}")

    final_weight_norm = model.diffusion_head_b.linear.weight.detach().float().norm().item()
    final_bias_norm = model.diffusion_head_b.linear.bias.detach().float().norm().item()
    print(
        "  Diffusion final layer norms: "
        f"weight={final_weight_norm:.6e}, bias={final_bias_norm:.6e}"
    )

    if not allow_partial and unexpected:
        raise RuntimeError(
            "Checkpoint has unexpected keys. "
            "Use --allow_partial_checkpoint only if you intentionally changed architecture."
        )
    if not allow_partial and missing and not base_loaded:
        raise RuntimeError(
            "Checkpoint is missing keys and no base Show-o2 checkpoint was loaded. "
            "Pass --showo_path /path/to/show-o2-1.5B or use a full unwrapped_model checkpoint."
        )
    if not allow_partial and critical_missing:
        raise RuntimeError(
            "Stage checkpoint is missing generation-critical modules. "
            "This would leave the tactile generator at base/random weights."
        )
    if final_weight_norm == 0.0 and final_bias_norm == 0.0:
        raise RuntimeError(
            "diffusion_head_b.linear is still exactly zero after loading. "
            "This usually means the fine-tuned Show-o2 checkpoint was not loaded, "
            "and ODE sampling will decode pure noise."
        )


def prepare_tactile_gen_input_like_training(
        prompts,
        text_tokenizer,
        showo_token_ids,
        num_visual_tokens,
        num_tactile_tokens,
        max_text_len,
        device,
):
    """
    Build inference tokens with the same sequence formatter as training:
    [BOS] text [BOV] visual [EOV] [BOV] tactile [EOV] [EOS] [PAD]...
    """
    batch_text_tokens = []
    batch_text_tokens_null = []
    batch_modality_positions = []
    batch_modality_positions_null = []

    for prompt in prompts:
        text_ids = text_tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_text_len,
        ).input_ids
        null_text_ids = text_tokenizer(
            "",
            add_special_tokens=False,
            truncation=True,
            max_length=max_text_len,
        ).input_ids

        text_tokens, _, modality_positions, _, _ = format_sequence_tactile_gen(
            text_tokens=text_ids,
            bos_id=showo_token_ids["bos_id"],
            eos_id=showo_token_ids["eos_id"],
            bov_id=showo_token_ids["bov_id"],
            eov_id=showo_token_ids["eov_id"],
            pad_id=text_tokenizer.pad_token_id,
            vid_pad_id=showo_token_ids["vid_pad_id"],
            num_visual_tokens=num_visual_tokens,
            num_tactile_tokens=num_tactile_tokens,
            max_seq_len=DEFAULT_SEQ_LEN,
        )
        text_tokens_null, _, modality_positions_null, _, _ = format_sequence_tactile_gen(
            text_tokens=null_text_ids,
            bos_id=showo_token_ids["bos_id"],
            eos_id=showo_token_ids["eos_id"],
            bov_id=showo_token_ids["bov_id"],
            eov_id=showo_token_ids["eov_id"],
            pad_id=text_tokenizer.pad_token_id,
            vid_pad_id=showo_token_ids["vid_pad_id"],
            num_visual_tokens=num_visual_tokens,
            num_tactile_tokens=num_tactile_tokens,
            max_seq_len=DEFAULT_SEQ_LEN,
        )

        batch_text_tokens.append(text_tokens)
        batch_text_tokens_null.append(text_tokens_null)
        batch_modality_positions.append(modality_positions)
        batch_modality_positions_null.append(modality_positions_null)

    return (
        torch.stack(batch_text_tokens, dim=0).to(device),
        torch.stack(batch_text_tokens_null, dim=0).to(device),
        torch.stack(batch_modality_positions, dim=0).to(device),
        torch.stack(batch_modality_positions_null, dim=0).to(device),
    )


def select_frame_indices(
        indices,
        num_frames,
        sampling_mode="contiguous",
        clip_start=0,
):
    if len(indices) == 0:
        raise ValueError("No frame indices available for sampling")

    ordered_indices = sorted(indices)

    if len(ordered_indices) >= num_frames:
        if sampling_mode == "contiguous":
            max_start = len(ordered_indices) - num_frames
            if clip_start < 0:
                start = max(max_start + clip_start + 1, 0)
            else:
                start = min(clip_start, max_start)
            end = start + num_frames
            return ordered_indices[start:end]

        selected_positions = np.linspace(0, len(ordered_indices) - 1, num_frames, dtype=int).tolist()
        return [ordered_indices[pos] for pos in selected_positions]

    return ordered_indices + [ordered_indices[-1]] * (num_frames - len(ordered_indices))


def load_video_frames(
        frame_dir,
        num_frames,
        image_size=DEFAULT_RESOLUTION,
        frame_indices=None,
        sampling_mode="contiguous",
        clip_start=0,
        return_indices=False,
):
    """
    从目录中均匀采样加载 PNG 帧。

    Args:
        frame_dir: 包含帧 PNG 文件的目录 (如 0.png, 1.png, ...)
        num_frames: 采样帧数
        image_size: 目标正方形分辨率
        frame_indices: 指定的帧号列表；若为空则从整个目录采样
        sampling_mode: contiguous 或 uniform
        clip_start: 连续采样模式下的起始偏移
        return_indices: 是否额外返回实际采样到的帧号

    Returns:
        Tensor of shape (num_frames, 3, image_size, image_size), 值域 [-1, 1]
        若 return_indices=True，则同时返回采样帧号
    """
    from torchvision import transforms

    frame_map = {
        int(os.path.splitext(filename)[0]): filename
        for filename in os.listdir(frame_dir)
        if filename.endswith(".png") and os.path.splitext(filename)[0].isdigit()
    }

    if len(frame_map) == 0:
        raise ValueError(f"No PNG files found in {frame_dir}")

    available_indices = sorted(frame_map)
    if frame_indices is None:
        selected = select_frame_indices(
            available_indices,
            num_frames,
            sampling_mode=sampling_mode,
            clip_start=clip_start,
        )
    else:
        filtered_indices = [idx for idx in frame_indices if idx in frame_map]
        if len(filtered_indices) == 0:
            raise ValueError(f"No requested frame indices exist in {frame_dir}")
        selected = select_frame_indices(
            filtered_indices,
            num_frames,
            sampling_mode=sampling_mode,
            clip_start=clip_start,
        )

    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    frames = []
    for idx in selected:
        frame_path = os.path.join(frame_dir, frame_map[idx])
        frame = Image.open(frame_path).convert('RGB')
        frames.append(transform(frame))

    stacked = torch.stack(frames, dim=0)
    if return_indices:
        return stacked, selected
    return stacked  # (T, C, H, W)


def save_video_frames(frames_tensor, output_path, fps=2):
    """
    保存生成的帧为 MP4 视频或 PNG 序列。

    Args:
        frames_tensor: (T, C, H, W) tensor in [-1, 1] range.
        output_path: 输出路径 (.mp4 或目录)
        fps: 帧率
    """
    output_dir = os.path.dirname(output_path)
    output_ext = os.path.splitext(output_path)[1].lower()

    if output_ext in {".mp4", ".gif", ".avi", ".mov"}:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        try:
            import imageio
            frames = (frames_tensor.float().permute(0, 2, 3, 1).cpu().numpy() + 1.0) / 2.0
            frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
            imageio.mimsave(output_path, frames, fps=fps)
            print(f"Saved generated tactile video to {output_path}")
            return
        except ImportError:
            print("imageio not installed. Saving as individual PNG frames instead.")
            output_path = os.path.splitext(output_path)[0]

    os.makedirs(output_path, exist_ok=True)
    frames_np = denorm_vid(frames_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4))[0]
    for i, frame in enumerate(frames_np):
        Image.fromarray(frame).save(os.path.join(output_path, f"frame_{i:04d}.png"))
    print(f"Saved {len(frames_np)} frames to {output_path}/")


def build_test_dataset(
        data_root,
        csv_path,
        text_tokenizer,
        showo_token_ids,
        num_frames,
        image_size,
        num_segment_tokens=None,
        split="test",
):
    return TactileVisualDataset(
        data_root=data_root,
        csv_path=csv_path,
        text_tokenizer=text_tokenizer,
        max_seq_len=DEFAULT_SEQ_LEN,
        image_size=image_size,
        latent_height=DEFAULT_LATENT_H,
        latent_width=DEFAULT_LATENT_W,
        num_frames=num_frames,
        num_visual_tokens_per_frame=DEFAULT_TOKENS_PER_FRAME,
        num_tactile_tokens_per_frame=DEFAULT_TOKENS_PER_FRAME,
        num_visual_tokens=num_segment_tokens,
        num_tactile_tokens=num_segment_tokens,
        cond_dropout_prob=0.0,
        split=split,
        frame_split_mode="contact_90_10",
        showo_token_ids=showo_token_ids,
    )


def run_single_inference(args, model, vae_model, text_tokenizer, showo_token_ids, device, weight_type):
    print(f"[4/4] Loading visual frames from {args.visual_video_dir}...")
    visual_frames = load_video_frames(
        args.visual_video_dir,
        args.num_frames,
        args.image_size,
        sampling_mode=args.sampling_mode,
        clip_start=args.clip_start,
    )
    print(f"  Loaded {visual_frames.shape[0]} frames at {args.image_size}x{args.image_size}")

    print("Generating tactile video...")
    print(f"  Text: {args.text[:100]}...")
    print(f"  Steps: {args.num_inference_steps}, CFG: {args.guidance_scale}")
    print(f"  Frames: {args.num_frames}")

    generated = generate_tactile_video(
        model=model,
        vae_model=vae_model,
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        text_prompt=args.text,
        visual_video_frames=visual_frames,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        device=device,
        weight_type=weight_type,
        time_embed_layout=args.time_embed_layout,
        sampling_method=args.sampling_method,
        atol=args.atol,
        rtol=args.rtol,
        reverse=args.reverse,
        time_shifting_factor=args.time_shifting_factor,
        vae_deterministic=args.vae_deterministic,
    )

    save_video_frames(generated, args.output_path, fps=args.fps)
    print("Done!")


def run_batch_test_inference(args, model, vae_model, text_tokenizer, showo_token_ids, device, weight_type):
    print(f"[4/4] Building test dataset from {args.tactile_csv_path}...")
    num_latent_tokens, num_segment_tokens, max_text_len, add_time_token = resolve_video_token_layout(
        model, args.num_frames, args.time_embed_layout
    )
    print(
        f"  Token layout: latent={num_latent_tokens}, segment={num_segment_tokens}, "
        f"add_time_token={add_time_token}, max_text_len={max_text_len}"
    )
    dataset = build_test_dataset(
        data_root=args.tactile_data_root,
        csv_path=args.tactile_csv_path,
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        num_frames=args.num_frames,
        image_size=args.image_size,
        num_segment_tokens=num_segment_tokens,
        split=args.eval_split,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")

    success_count = 0
    with open(manifest_path, "w") as manifest_file:
        for sample_idx, sample in enumerate(dataset.samples, start=1):
            object_name = sample["object_name"]
            output_path = os.path.join(args.output_dir, f"{object_name}.mp4")
            print(f"[{sample_idx}/{len(dataset)}] Generating {object_name}...")

            record = {
                "object_name": object_name,
                "text": sample["text"],
                "split_frame_indices": sample["frame_indices"],
                "output_path": output_path,
            }

            try:
                selected_indices = dataset._select_frame_indices(sample["frame_indices"])
                visual_frames, sampled_frame_indices = dataset.load_sample_video(
                    sample, "visual", selected_indices
                )
                tactile_target, _ = dataset.load_sample_video(
                    sample, "tactile", selected_indices
                )
                generated = generate_tactile_video(
                    model=model,
                    vae_model=vae_model,
                    text_tokenizer=text_tokenizer,
                    showo_token_ids=showo_token_ids,
                    text_prompt=sample["text"],
                    visual_video_frames=visual_frames,
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    device=device,
                    weight_type=weight_type,
                    time_embed_layout=args.time_embed_layout,
                    sampling_method=args.sampling_method,
                    atol=args.atol,
                    rtol=args.rtol,
                    reverse=args.reverse,
                    time_shifting_factor=args.time_shifting_factor,
                    vae_deterministic=args.vae_deterministic,
                )
                save_video_frames(generated, output_path, fps=args.fps)
                if args.save_conditions:
                    stem = os.path.splitext(output_path)[0]
                    save_video_frames(visual_frames, f"{stem}_visual.mp4", fps=args.fps)
                    save_video_frames(tactile_target, f"{stem}_target.mp4", fps=args.fps)
                record["sampled_frame_indices"] = sampled_frame_indices
                record["status"] = "ok"
                success_count += 1
            except Exception as exc:
                print(f"  Failed on {object_name}: {exc}")
                record["status"] = "error"
                record["error"] = str(exc)

            manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest_file.flush()

    print(f"Batch inference finished: {success_count}/{len(dataset)} samples succeeded.")
    print(f"Manifest saved to {manifest_path}")


@torch.no_grad()
def generate_tactile_video(
        model,
        vae_model,
        text_tokenizer,
        showo_token_ids,
        text_prompt: str,
        visual_video_frames: torch.Tensor,
        num_frames: int = DEFAULT_NUM_FRAMES,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        device: str = 'cuda',
        weight_type: torch.dtype = torch.bfloat16,
        time_embed_layout: str = "auto",
        sampling_method: str = "euler",
        atol: float = 1e-6,
        rtol: float = 1e-3,
        reverse: bool = False,
        time_shifting_factor: float = 3.0,
        vae_deterministic: bool = False,
):
    """
    基于文本 + 视觉视频条件生成触觉视频。

    算法流程:
        1. VAE 编码视觉视频帧 → visual_latents (干净条件, t=1.0)
        2. 随机初始化触觉 latent 噪声
        3. 构建序列: [BOS]text[BOV]visual[EOV][BOV]noise[EOV]
        4. Flow-Matching ODE 去噪触觉段
        5. VAE 解码 → 触觉视频帧

    Args:
        model: Showo2Qwen2_5 模型 (阶段一训练后)
        vae_model: WanVAE 编码/解码器
        text_tokenizer: HuggingFace tokenizer
        showo_token_ids: 特殊 token ID 字典
        text_prompt: 物理先验文本描述
        visual_video_frames: (T, C, H, W) 视觉视频帧
        num_frames: 生成帧数
        num_inference_steps: ODE 求解器步数
        guidance_scale: CFG 引导强度 (0=无引导)
        device: 计算设备
        weight_type: 模型精度

    Returns:
        Generated tactile video frames as (T, C, H, W) tensor in [-1, 1].
    """
    model.eval()

    # ---- 计算 token 数量 ----
    # The model consumes one extra position per visual region when add_time_embeds=True:
    # [time_embed] + latent patch tokens. Keep this aligned with the checkpoint config.
    num_latent_tokens, num_segment_tokens, max_text_len, add_time_token = resolve_video_token_layout(
        model, num_frames, time_embed_layout
    )

    # ---- 1. VAE 编码视觉视频帧 ----
    # 与训练保持一致: clip-level 编码 (B, C, T, H, W)，使用 WanVAE 3D causal encoder
    # 训练代码: pixel_values shape (B*2, C, num_frames, H, W) -> vae_model.sample()
    # 训练数据流: (B,2,T,C,H,W) -> rearrange -> (B*2,C,T,H,W)
    # 推理数据流: (T,C,H,W) -> rearrange -> (1,C,T,H,W)
    visual_pixels = visual_video_frames.to(device).to(weight_type)  # (T, C, H, W)
    visual_pixels = rearrange(visual_pixels, 't c h w -> 1 c t h w')  # (1, C, T, H, W)

    visual_latents = vae_model.sample(
        visual_pixels,
        deterministic=vae_deterministic,
    )  # (1, 16, T_vis, Hv, Wv) — T_vis 由 VAE temporal compression 决定
    if visual_latents.shape[2] == 1:
        visual_latents = visual_latents.squeeze(2)  # (1, 16, Hv, Wv) — 单帧特例
    # visual_latents is now (1, C, T_vis, Hv, Wv) where T_vis = VAE output temporal dim
    _, c, T_vis, hv, wv = visual_latents.shape
    patch_size = int(getattr(model.config, "patch_size", 2))
    actual_latent_tokens = T_vis * (hv // patch_size) * (wv // patch_size)

    # ---- 如果 VAE 时间压缩导致 T_vis != num_frames，动态调整 token layout ----
    if T_vis != num_frames:
        print(
            f"  Note: VAE temporal compression: input {num_frames} frames → {T_vis} latent frames"
        )
        if actual_latent_tokens != num_latent_tokens:
            print(
                f"  Adjusting token layout: {num_latent_tokens} → {actual_latent_tokens} per segment"
            )
            # Recompute layout based on actual VAE output (matching training behavior)
            num_latent_tokens = actual_latent_tokens
            num_segment_tokens = num_latent_tokens + int(add_time_token)
            max_text_len = DEFAULT_SEQ_LEN - 2 * num_segment_tokens - 6
            if max_text_len <= 0:
                raise ValueError(
                    f"Sequence too short after VAE temporal compression adjustment: "
                    f"max_seq_len={DEFAULT_SEQ_LEN}, segment_tokens={num_segment_tokens}"
                )

    # ---- 2. 构建序列 ----
    batch_text_tokens, batch_text_tokens_null, \
        batch_modality_positions, batch_modality_positions_null = \
        prepare_tactile_gen_input_like_training(
            prompts=[text_prompt],
            text_tokenizer=text_tokenizer,
            showo_token_ids=showo_token_ids,
            num_visual_tokens=num_segment_tokens,
            num_tactile_tokens=num_segment_tokens,
            max_text_len=max_text_len,
            device=device,
        )

    actual_seq_len = batch_text_tokens.size(1)
    print(
        f"  Token layout: latent={num_latent_tokens}, segment={num_segment_tokens}, "
        f"add_time_token={add_time_token}, max_text_len={max_text_len}, seq_len={actual_seq_len}"
    )

    # ---- 3. 初始化触觉侧随机噪声 ----
    # 噪声维度必须与 VAE latent 输出一致 (T_vis 而非 num_frames)
    z_tactile = torch.randn(
        (1, c, T_vis, hv, wv),
        device=device, dtype=weight_type
    )

    # ---- 4. 拼接: visual(干净) + tactile(噪声) ----
    # 与训练保持一致: 一个样本有两个视频段 (visual, tactile)
    image_latents = torch.cat([visual_latents, z_tactile], dim=0)  # (2, C, T, Hv, Wv)

    # ---- 5. 构建 Omni-Attention 掩码 ----
    if guidance_scale > 0:
        text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
        modality_positions = torch.cat(
            [batch_modality_positions, batch_modality_positions_null], dim=0
        )
    else:
        text_tokens = batch_text_tokens
        modality_positions = batch_modality_positions

    block_mask = omni_attn_mask_naive(
        text_tokens.size(0), actual_seq_len, modality_positions, device
    ).to(weight_type)

    # ---- 6. Flow Matching 传输 & ODE 采样器 ----
    transport = create_transport(
        path_type="Linear",
        prediction="velocity",
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
        snr_type="lognorm",
        do_shift=True,
        seq_len=num_segment_tokens,
    )
    sampler = Sampler(transport)

    # ---- 7. CFG: 复制两个视频段给无条件分支 ----
    if guidance_scale > 0:
        initial_latents = torch.cat([image_latents, image_latents], dim=0)  # (4, C, T, Hv, Wv)
    else:
        initial_latents = image_latents  # (2, C, T, Hv, Wv)

    # ---- 8. 去噪参数 ----
    model_kwargs = dict(
        text_tokens=text_tokens,
        attention_mask=block_mask,
        modality_positions=modality_positions,
        output_hidden_states=True,
        max_seq_len=actual_seq_len,
        guidance_scale=guidance_scale if guidance_scale > 0 else 0.0,
        only_denoise_last_image=True,  # 只去噪触觉段 (最后一个视觉区域)
    )

    # ---- 9. ODE 去噪 ----
    sample_fn = sampler.sample_ode(
        sampling_method=sampling_method,
        num_steps=num_inference_steps,
        atol=atol,
        rtol=rtol,
        reverse=reverse,
        time_shifting_factor=time_shifting_factor,
    )

    with autocast_context(device, weight_type):
        samples = sample_fn(initial_latents, model.t2i_generate, **model_kwargs)[-1]

    if guidance_scale > 0:
        samples = torch.chunk(samples, 2)[0]

    # 只解码最后一个视频段，也就是 tactile 段
    samples = samples[-1:]

    # ---- 10. VAE 解码触觉 latent → 像素空间 ----
    if samples.dim() == 3:
        # 模型输出 patchified 格式，需要 unpatchify
        p = 2  # patch_size
        samples = rearrange(
            samples, 'b (t h w) (p1 p2 c) -> b c t (h p1) (w p2)',
            t=T_vis, h=hv, w=wv, p1=p, p2=p, c=c
        )

    # 与训练保持一致: clip-level VAE 解码
    # 训练代码: recons_images = vae_model.batch_decode(image_latents)
    # image_latents shape: (B, C, T_latent, Hv, Wv) -> batch_decode -> (B, 3, T_latent, H, W)
    generated_frames = vae_model.batch_decode(samples)  # (1, 3, T_vis, H, W)
    generated_frames = generated_frames.squeeze(0)  # (3, T_vis, H, W)
    generated_frames = rearrange(generated_frames, 'c t h w -> t c h w')  # (T_vis, 3, H, W)

    return generated_frames


def main():
    parser = argparse.ArgumentParser(
        description="Tactile Video Generation — Stage 1 Inference"
    )

    # ---- 模型路径 (集群本地) ----
    parser.add_argument("--stage1_checkpoint", type=str, required=True,
                        help="阶段一训练输出的 checkpoint 路径")
    parser.add_argument("--vae_path", type=str, required=True,
                        help="WanVAE 权重本地路径 (Wan2.1_VAE.pth)")
    parser.add_argument("--llm_path", type=str, required=True,
                        help="Qwen2.5 Tokenizer 本地路径")
    parser.add_argument("--showo_path", type=str, default=None,
                        help="Show-o2 预训练权重本地路径 (仅当 stage1_checkpoint 不完整时需要)")
    parser.add_argument("--siglip_path", type=str, default=None,
                        help="SigLIP 权重本地路径 (如 /defaultShare/models/siglip-so400m-patch14-384)")

    # ---- 输入 ----
    parser.add_argument("--text", type=str, default=None,
                        help="物理文本描述")
    parser.add_argument("--visual_video_dir", type=str, default=None,
                        help="视觉视频帧目录 (img_gelsight/)")
    parser.add_argument("--batch_test", action="store_true",
                        help="使用 test split 进行批量推理")
    parser.add_argument("--eval_split", type=str, default="test",
                        choices=["train", "test", "all"],
                        help="批量推理使用的数据划分；train 可复现训练验证生成难度")
    parser.add_argument("--tactile_data_root", type=str, default=None,
                        help="触觉数据根目录，用于批量 test 推理")
    parser.add_argument("--tactile_csv_path", type=str, default=None,
                        help="触觉 CSV 路径，用于批量 test 推理")

    # ---- 输出 ----
    parser.add_argument("--output_path", type=str, default="output_tactile.mp4",
                        help="生成的触觉视频输出路径 (.mp4 或目录)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="批量 test 推理输出目录")
    parser.add_argument("--fps", type=int, default=2,
                        help="输出视频 fps；训练 wandb 验证视频使用 2")
    parser.add_argument("--save_conditions", action="store_true",
                        help="批量推理时同时保存 visual condition 和 tactile target 视频")

    # ---- 生成参数 ----
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES,
                        help="生成帧数")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="ODE 求解器步数 (越多质量越好, 越慢)")
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                        help="CFG 文本引导强度 (0=无引导)")
    parser.add_argument("--sampling_method", type=str, default="euler",
                        help="ODE sampling method, e.g. euler or dopri5")
    parser.add_argument("--atol", type=float, default=1e-6,
                        help="ODE solver absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-3,
                        help="ODE solver relative tolerance")
    parser.add_argument("--reverse", action="store_true",
                        help="Reverse ODE integration interval")
    parser.add_argument("--time_shifting_factor", type=float, default=3.0,
                        help="ODE time shifting factor used by the official Show-o2 sampler")
    parser.add_argument("--image_size", type=int, default=DEFAULT_RESOLUTION,
                        help="帧 resize 分辨率")
    parser.add_argument("--sampling_mode", type=str, default="contiguous",
                        choices=["contiguous", "uniform"],
                        help="单样本推理的取帧方式，默认连续采样")
    parser.add_argument("--clip_start", type=int, default=0,
                        help="单样本连续采样时的起始偏移，默认从序列开头开始")
    parser.add_argument("--time_embed_layout", type=str, default="auto",
                        choices=["auto", "with_time_token", "without_time_token"],
                        help="视频 token 段是否包含 time embedding 位置；auto 根据 checkpoint config 判断")
    parser.add_argument("--virtual_force_coeff", type=float, default=0.0,
                        help="训练阶段 virtual force proxy loss 系数；推理阶段仅用于记录/脚本兼容")
    parser.add_argument("--contact_weighted_flow_alpha", type=float, default=0.0,
                        help="训练阶段 contact-weighted flow loss 系数；推理阶段仅用于记录/脚本兼容")
    parser.add_argument("--vae_deterministic", action="store_true",
                        help="使用 VAE posterior mean 编码 visual condition，便于排除 VAE 随机采样误差")
    parser.add_argument("--allow_partial_checkpoint", action="store_true",
                        help="允许 checkpoint missing/unexpected keys；默认遇到生成关键模块不完整会报错")

    # ---- 设备 ----
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备")
    args = parser.parse_args()

    # ========================================
    # 设置离线模式环境变量
    # ========================================
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    weight_type = torch.bfloat16
    print(
        "Inference provenance: "
        f"virtual_force_coeff={args.virtual_force_coeff}, "
        f"contact_weighted_flow_alpha={args.contact_weighted_flow_alpha}"
    )

    if args.batch_test:
        required_args = {
            "--tactile_data_root": args.tactile_data_root,
            "--tactile_csv_path": args.tactile_csv_path,
            "--output_dir": args.output_dir,
        }
    else:
        required_args = {
            "--text": args.text,
            "--visual_video_dir": args.visual_video_dir,
        }

    missing_args = [name for name, value in required_args.items() if not value]
    if missing_args:
        parser.error(f"Missing required arguments: {', '.join(missing_args)}")

    # ========================================
    # 1. 加载 VAE
    # ========================================
    print(f"[1/4] Loading VAE from {args.vae_path}...")
    vae_model = WanVAE(
        vae_pth=args.vae_path,
        dtype=weight_type, device=device
    )
    print("  VAE loaded.")

    # ========================================
    # 2. 加载 Tokenizer
    # ========================================
    print(f"[2/4] Loading Tokenizer from {args.llm_path}...")
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        args.llm_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[args.llm_path]
    )
    print(f"  Tokenizer loaded. vocab_size={len(text_tokenizer)}")

    # ========================================
    # 3. 加载模型 (从阶段一 checkpoint)
    # ========================================
    # 不能用 from_pretrained() 因为它会读取 checkpoint 的 config.json，
    # 其中 llm_model_path / clip_pretrained_model_path 指向旧集群路径，
    # 在新集群上不存在，导致 AutoConfig.from_pretrained() 失败。
    # 改为: 用 CLI 传入的正确本地路径构建模型，再手动加载 state_dict。
    stage1_checkpoint = resolve_checkpoint_dir(args.stage1_checkpoint)
    print(f"[3/4] Loading model from {stage1_checkpoint}...")

    # 3a. Read checkpoint config.json for architecture parameters
    config_path = os.path.join(stage1_checkpoint, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {stage1_checkpoint}")
    with open(config_path, 'r') as f:
        checkpoint_config = json.load(f)

    # 3b. Override path-dependent fields with CLI args (new cluster paths)
    checkpoint_config['llm_model_path'] = args.llm_path
    if not os.path.isdir(args.llm_path):
        print(f"  Warning: --llm_path '{args.llm_path}' does not exist or is not a directory")

    if args.siglip_path:
        checkpoint_config['clip_pretrained_model_path'] = args.siglip_path
    else:
        # 如果没传 --siglip_path，检查 config 中的旧路径在当前机器是否可用
        old_siglip = checkpoint_config.get('clip_pretrained_model_path')
        if not old_siglip or not os.path.isdir(old_siglip):
            raise ValueError(
                f"Config contains clip_pretrained_model_path='{old_siglip}' "
                f"which does not exist on this machine.\n"
                f"Please pass --siglip_path with the correct local path, e.g.\n"
                f"  --siglip_path /defaultShare/models/siglip-so400m-patch14-384"
            )

    # load_from_showo=True: only reads config.json from llm_path (not weights),
    # because all weights come from the stage-1 checkpoint state_dict.
    checkpoint_config['load_from_showo'] = True

    # 3c. Construct model with corrected local paths
    # Keep Show-o2 weights in fp32, matching train_tactile_stage_one.py.
    # VAE inputs/latents and attention masks still use weight_type below.
    model = Showo2Qwen2_5(**checkpoint_config).to(device)

    base_loaded = False
    if args.showo_path:
        base_weight_file = resolve_weight_file(args.showo_path)
        if base_weight_file is None:
            raise FileNotFoundError(
                f"Base Show-o2 weights not found in --showo_path={args.showo_path}"
            )
        print(f"  Loading base Show-o2 weights from {base_weight_file}...")
        base_state_dict = load_checkpoint_state_dict(base_weight_file)
        base_missing, base_unexpected = model.load_state_dict(base_state_dict, strict=False)
        print(
            f"  Base load: missing_keys={len(base_missing)}, "
            f"unexpected_keys={len(base_unexpected)}"
        )
        if base_missing:
            print(f"  Base missing first keys: {base_missing[:20]}")
        if base_unexpected:
            print(f"  Base unexpected first keys: {base_unexpected[:20]}")
        del base_state_dict
        base_loaded = True

    # 3d. Load fine-tuned weights from stage-1 checkpoint
    weight_file = resolve_weight_file(stage1_checkpoint)
    if weight_file is None:
        raise FileNotFoundError(
            f"Model weights not found in {stage1_checkpoint} "
            f"(tried pytorch_model.bin and model.safetensors)"
        )
    print(f"  Loading state_dict from {weight_file}...")
    state_dict = load_checkpoint_state_dict(weight_file)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: missing keys: {missing[:5]}..." if len(missing) > 5
              else f"  Warning: missing keys: {missing}")
    if unexpected:
        print(f"  Warning: unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5
              else f"  Warning: unexpected keys: {unexpected}")
    check_checkpoint_load(
        model,
        missing,
        unexpected,
        allow_partial=args.allow_partial_checkpoint,
        base_loaded=base_loaded,
    )
    del state_dict

    model.eval()
    print("  Model loaded.")

    if args.batch_test:
        run_batch_test_inference(
            args, model, vae_model, text_tokenizer, showo_token_ids, device, weight_type
        )
    else:
        run_single_inference(
            args, model, vae_model, text_tokenizer, showo_token_ids, device, weight_type
        )


if __name__ == "__main__":
    main()
