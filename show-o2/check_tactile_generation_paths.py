# coding=utf-8
"""
Offline two-path tactile generation check for the A100 cluster.

Path A simulates the training validation generation path for one fixed dataset
sample. Path B simulates the standalone inference path on the same visual frames.
Both paths share the same tactile noise so differences are easier to localize.
"""

import argparse
import json
import os
import random
from contextlib import nullcontext
from typing import Dict

import numpy as np
import torch
from einops import rearrange

from datasets import TactileVisualDataset
from datasets.utils import format_sequence_tactile_gen
from inference_tactile_video import (
    DEFAULT_LATENT_H,
    DEFAULT_LATENT_W,
    DEFAULT_NUM_FRAMES,
    DEFAULT_RESOLUTION,
    DEFAULT_SEQ_LEN,
    DEFAULT_TOKENS_PER_FRAME,
    check_checkpoint_load,
    load_checkpoint_state_dict,
    load_video_frames,
    prepare_tactile_gen_input_like_training,
    resolve_checkpoint_dir,
    resolve_video_token_layout,
    resolve_weight_file,
    save_video_frames,
)
from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
from models.misc import get_text_tokenizer
from transport import Sampler, create_transport
from utils import path_to_llm_name


DEFAULT_MODEL_ROOT = "/defaultShare/models"
DEFAULT_DATA_ROOT = "/defaultShare/data_indoor"
DEFAULT_CSV_PATH = "/18009672469/xy/Show-o/show-o2/contact_indoor_list_tvl.csv"
DEFAULT_CHECKPOINT = (
    "/18009672469/xy/Show-o/show-o2/outputs/"
    "showo2-1.5b-tactile-stage-1/checkpoint-18000"
)
DEFAULT_VISUAL_DIR = "/defaultShare/data_indoor/3dprint/img_gelsight"
DEFAULT_OUTPUT_DIR = (
    "/18009672469/xy/Show-o/show-o2/Inference/"
    "path_check_3dprint_ckpt18000"
)
DEFAULT_TEXT = (
    "The touch of Gadget Holder is smooth, moderate roughness, hardness, "
    "akin to flexible rubber, slightly matte yet waxy texture, it is made of rubber."
)
STAT_KEYS = ("mean", "std", "norm", "min", "max", "finite_ratio")
COMPARE_KEYS = (
    "selected_indices",
    "text_tokens_first_64",
    "modality_positions",
    "visual_frames",
    "visual_latents",
    "target_tactile_frames",
    "target_tactile_latents",
    "shared_z_tactile",
    "initial_latents",
    "block_mask",
    "v_pred_t=0.0",
    "v_pred_t=0.5",
    "v_pred_t=1.0",
    "ode_samples_raw_final",
    "ode_samples_after_cfg_chunk",
    "generated_tactile_latents",
    "generated_vs_target_latent_l1",
    "generated_vs_target_latent_mse",
    "generated_frames",
    "generated_vs_target_frame_l1",
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device, dtype):
    if device.type == "cuda" and dtype in (torch.bfloat16, torch.float16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def tensor_stats(tensor, max_elements=4096, first_values=8):
    if tensor is None:
        return None

    with torch.no_grad():
        data = tensor.detach()
        flat = data.reshape(-1)
        numel = flat.numel()
        if numel > max_elements:
            indices = torch.arange(max_elements, device=flat.device, dtype=torch.long)
            indices = indices * (numel - 1) // (max_elements - 1)
            stats_flat = flat[indices]
            sampled = True
        else:
            stats_flat = flat
            sampled = False

        stats_float = stats_flat.float()
        finite = torch.isfinite(stats_float)
        finite_values = stats_float[finite]
        if finite_values.numel() == 0:
            numeric_stats = {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "norm": None,
                "finite_ratio": 0.0,
            }
        else:
            numeric_stats = {
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
                "mean": float(finite_values.mean().item()),
                "std": float(finite_values.std(unbiased=False).item()),
                "norm": float(finite_values.norm().item()),
                "finite_ratio": float(finite.float().mean().item()),
            }

        return {
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "device": str(data.device),
            "numel": int(numel),
            "sampled": sampled,
            "sample_size": int(stats_flat.numel()),
            "first_values": flat[:min(first_values, numel)].float().cpu().tolist(),
            **numeric_stats,
        }


def write_trace(path: str, title: str, rows: Dict[str, object]):
    with open(path, "w", encoding="utf-8") as trace_file:
        trace_file.write(f"===== {title} =====\n")
        for key, value in rows.items():
            trace_file.write(
                f"{key}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}\n"
            )


def prepare_training_generation_batch(
        prompts,
        text_tokenizer,
        showo_token_ids,
        num_visual_tokens,
        num_tactile_tokens,
        max_text_len,
        device,
):
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


def load_models(args, device, weight_type):
    model_root = args.model_root
    vae_path = os.path.join(model_root, "Wan2.1_VAE.pth")
    llm_path = os.path.join(model_root, "Qwen2.5-1.5B-Instruct")
    showo_path = os.path.join(model_root, "show-o2-1.5B")
    siglip_path = os.path.join(model_root, "siglip-so400m-patch14-384")

    print(f"[1/4] Loading VAE from {vae_path}...")
    vae_model = WanVAE(vae_pth=vae_path, dtype=weight_type, device=device)

    print(f"[2/4] Loading tokenizer from {llm_path}...")
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        llm_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[llm_path],
    )

    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint)
    print(f"[3/4] Loading model from {checkpoint_dir}...")
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {checkpoint_dir}")
    with open(config_path, "r", encoding="utf-8") as config_file:
        checkpoint_config = json.load(config_file)

    checkpoint_config["llm_model_path"] = llm_path
    checkpoint_config["clip_pretrained_model_path"] = siglip_path
    checkpoint_config["load_from_showo"] = True
    model = Showo2Qwen2_5(**checkpoint_config).to(device)
    if args.cast_model_bf16:
        model = model.to(weight_type)

    base_weight_file = resolve_weight_file(showo_path)
    if base_weight_file is None:
        raise FileNotFoundError(f"Base Show-o2 weights not found in {showo_path}")
    print(f"  Loading base Show-o2 weights from {base_weight_file}...")
    base_state_dict = load_checkpoint_state_dict(base_weight_file)
    base_missing, base_unexpected = model.load_state_dict(base_state_dict, strict=False)
    print(
        f"  Base load: missing_keys={len(base_missing)}, "
        f"unexpected_keys={len(base_unexpected)}"
    )
    del base_state_dict

    weight_file = resolve_weight_file(checkpoint_dir)
    if weight_file is None:
        raise FileNotFoundError(f"Model weights not found in {checkpoint_dir}")
    print(f"  Loading stage checkpoint weights from {weight_file}...")
    state_dict = load_checkpoint_state_dict(weight_file)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    check_checkpoint_load(
        model,
        missing,
        unexpected,
        allow_partial=False,
        base_loaded=True,
    )
    del state_dict

    model.eval()
    return model, vae_model, text_tokenizer, showo_token_ids


def build_training_dataset(args, text_tokenizer, showo_token_ids, num_segment_tokens):
    return TactileVisualDataset(
        data_root=args.data_root,
        csv_path=args.csv_path,
        text_tokenizer=text_tokenizer,
        max_seq_len=DEFAULT_SEQ_LEN,
        image_size=args.image_size,
        latent_height=DEFAULT_LATENT_H,
        latent_width=DEFAULT_LATENT_W,
        num_frames=args.num_frames,
        num_visual_tokens_per_frame=DEFAULT_TOKENS_PER_FRAME,
        num_tactile_tokens_per_frame=DEFAULT_TOKENS_PER_FRAME,
        num_visual_tokens=num_segment_tokens,
        num_tactile_tokens=num_segment_tokens,
        cond_dropout_prob=0.0,
        split="train",
        frame_split_mode="contact_90_10",
        showo_token_ids=showo_token_ids,
    )


def find_object_sample(dataset, object_name):
    for sample in dataset.samples:
        if sample["object_name"] == object_name:
            return sample
    available = ", ".join(sample["object_name"] for sample in dataset.samples[:20])
    raise ValueError(f"Object {object_name!r} not found. First available samples: {available}")


def encode_visual_latents(
        vae_model,
        visual_frames,
        device,
        weight_type,
        deterministic,
):
    visual_pixels = visual_frames.to(device=device, dtype=weight_type).unsqueeze(2)
    visual_latents = vae_model.sample(
        visual_pixels,
        deterministic=deterministic,
    )
    if visual_latents.shape[2] != 1:
        raise ValueError(f"Expected temporal VAE size 1, got {visual_latents.shape}")
    visual_latents = visual_latents.squeeze(2)
    return rearrange(visual_latents, "t c h w -> 1 c t h w")


def encode_training_pair_latents(
        vae_model,
        visual_frames,
        tactile_frames,
        device,
        weight_type,
        deterministic,
):
    """Match train_tactile_stage_one.prepare_latents_and_labels VAE layout."""
    pixel_values = torch.stack([visual_frames, tactile_frames], dim=0).unsqueeze(0)
    num_frames = pixel_values.shape[2]
    pixel_values = rearrange(pixel_values, "b m t c h w -> (b m t) c h w")
    pixel_values = pixel_values.to(device=device, dtype=weight_type).unsqueeze(2)

    image_latents = vae_model.sample(pixel_values, deterministic=deterministic)
    if image_latents.shape[2] != 1:
        raise ValueError(f"Expected temporal VAE size 1, got {image_latents.shape}")
    image_latents = image_latents.squeeze(2)

    c, h, w = image_latents.shape[1:]
    image_latents = rearrange(
        image_latents.reshape(1, 2, num_frames, c, h, w),
        "b m t c h w -> (b m) c t h w",
    )
    return image_latents[0:1], image_latents[1:2]


def build_model_inputs(
        batch_text_tokens,
        batch_text_tokens_null,
        batch_modality_positions,
        batch_modality_positions_null,
        image_latents,
        guidance_scale,
        weight_type,
        device,
):
    if guidance_scale > 0:
        initial_latents = torch.cat([image_latents, image_latents], dim=0)
        text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
        modality_positions = torch.cat(
            [batch_modality_positions, batch_modality_positions_null], dim=0
        )
    else:
        initial_latents = image_latents
        text_tokens = batch_text_tokens
        modality_positions = batch_modality_positions

    block_mask = omni_attn_mask_naive(
        text_tokens.size(0),
        text_tokens.size(1),
        modality_positions,
        device,
    ).to(weight_type)
    return initial_latents, text_tokens, modality_positions, block_mask


@torch.no_grad()
def run_generation_path(
        label,
        model,
        vae_model,
        visual_frames,
        visual_latents,
        shared_z_tactile,
        token_batch,
        selected_indices,
        args,
        device,
        weight_type,
        target_tactile_frames=None,
        target_tactile_latents=None,
):
    (
        batch_text_tokens,
        batch_text_tokens_null,
        batch_modality_positions,
        batch_modality_positions_null,
    ) = token_batch

    image_latents = torch.cat([visual_latents, shared_z_tactile], dim=0)
    initial_latents, text_tokens, modality_positions, block_mask = build_model_inputs(
        batch_text_tokens=batch_text_tokens,
        batch_text_tokens_null=batch_text_tokens_null,
        batch_modality_positions=batch_modality_positions,
        batch_modality_positions_null=batch_modality_positions_null,
        image_latents=image_latents,
        guidance_scale=args.guidance_scale,
        weight_type=weight_type,
        device=device,
    )

    model_kwargs = dict(
        text_tokens=text_tokens,
        attention_mask=block_mask,
        modality_positions=modality_positions,
        output_hidden_states=True,
        max_seq_len=text_tokens.size(1),
        guidance_scale=args.guidance_scale if args.guidance_scale > 0 else 0.0,
        only_denoise_last_image=True,
    )

    trace = {
        "label": label,
        "model_param_dtype": str(next(model.parameters()).dtype),
        "selected_indices": selected_indices,
        "text": args.text,
        "text_tokens": tensor_stats(text_tokens),
        "text_tokens_first_64": text_tokens[0, :64].detach().cpu().tolist(),
        "modality_positions": modality_positions.detach().cpu().tolist(),
        "visual_frames": tensor_stats(visual_frames),
        "visual_latents": tensor_stats(visual_latents),
        "target_tactile_frames": tensor_stats(target_tactile_frames),
        "target_tactile_latents": tensor_stats(target_tactile_latents),
        "shared_z_tactile": tensor_stats(shared_z_tactile),
        "image_latents_visual_plus_tactile_noise": tensor_stats(image_latents),
        "initial_latents": tensor_stats(initial_latents),
        "block_mask": tensor_stats(block_mask),
    }

    for t_value in (0.0, 0.5, 1.0):
        t_probe = torch.full(
            (initial_latents.shape[0],),
            t_value,
            device=device,
            dtype=torch.float32,
        )
        with autocast_context(device, weight_type):
            v_probe = model.t2i_generate(initial_latents, t_probe, **model_kwargs)
        if args.guidance_scale > 0:
            v_probe = torch.chunk(v_probe, 2)[0]
        trace[f"v_pred_t={t_value}"] = tensor_stats(v_probe)

    transport = create_transport(
        path_type="Linear",
        prediction="velocity",
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
        snr_type="lognorm",
        do_shift=True,
        seq_len=args.num_segment_tokens,
    )
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method=args.sampling_method,
        num_steps=args.num_inference_steps,
        atol=args.atol,
        rtol=args.rtol,
        reverse=args.reverse,
        time_shifting_factor=args.time_shifting_factor,
    )
    with autocast_context(device, weight_type):
        samples = sample_fn(initial_latents, model.t2i_generate, **model_kwargs)[-1]
    trace["ode_samples_raw_final"] = tensor_stats(samples)

    if args.guidance_scale > 0:
        samples = torch.chunk(samples, 2)[0]
        trace["ode_samples_after_cfg_chunk"] = tensor_stats(samples)

    generated_tactile_latents = samples[-1:]
    trace["generated_tactile_latents"] = tensor_stats(generated_tactile_latents)
    if target_tactile_latents is not None:
        trace["generated_vs_target_latent_l1"] = float(
            (generated_tactile_latents.float() - target_tactile_latents.float())
            .abs()
            .mean()
            .item()
        )
        trace["generated_vs_target_latent_mse"] = float(
            (
                generated_tactile_latents.float()
                - target_tactile_latents.float()
            ).pow(2).mean().item()
        )

    generated_frames = vae_model.batch_decode(
        rearrange(generated_tactile_latents, "b c t h w -> (b t) c h w").unsqueeze(2)
    ).squeeze(2)
    trace["generated_frames"] = tensor_stats(generated_frames)
    if target_tactile_frames is not None:
        trace["generated_vs_target_frame_l1"] = float(
            (generated_frames.float().cpu() - target_tactile_frames.float())
            .abs()
            .mean()
            .item()
        )
    return generated_frames, trace


def compare_value(key, left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        lines = []
        shape_diff = left.get("shape") != right.get("shape")
        lines.append(
            f"{'DIFF' if shape_diff else 'OK'} {key}.shape: "
            f"A={left.get('shape')} B={right.get('shape')}"
        )
        diverged = shape_diff
        for stat_key in STAT_KEYS:
            lv = left.get(stat_key)
            rv = right.get(stat_key)
            if lv is None or rv is None:
                lines.append(f"NA {key}.{stat_key}: A={lv} B={rv}")
                continue
            abs_diff = abs(rv - lv)
            denom = max(abs(lv), 1e-12)
            rel_diff = abs_diff / denom
            if rel_diff > 1e-4 and abs_diff > 1e-6:
                diverged = True
            lines.append(
                f"STAT {key}.{stat_key}: A={lv:.6g} B={rv:.6g} "
                f"abs_diff={abs_diff:.6g} rel_diff={rel_diff:.6g}"
            )
        return diverged, "\n".join(lines)

    diverged = left != right
    return diverged, f"{'DIFF' if diverged else 'OK'} {key}: A={left} B={right}"


def write_compare_report(path, trace_a, trace_b):
    lines = [
        "Two-path tactile generation comparison",
        "",
        f"Path A: {trace_a.get('label')}",
        f"Path B: {trace_b.get('label')}",
        "",
    ]
    first_divergence = None
    for key in COMPARE_KEYS:
        if key not in trace_a or key not in trace_b:
            diverged = True
            block = (
                f"MISS {key}: "
                f"A={'yes' if key in trace_a else 'no'} "
                f"B={'yes' if key in trace_b else 'no'}"
            )
        else:
            diverged, block = compare_value(key, trace_a[key], trace_b[key])
        if diverged and first_divergence is None:
            first_divergence = key
        lines.append(block)
        lines.append("")

    lines.insert(
        4,
        f"First obvious divergence: {first_divergence or 'none within checked fields'}",
    )
    with open(path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Offline tactile path A/B checker.")
    parser.add_argument("--model_root", type=str, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH)
    parser.add_argument("--visual_dir", type=str, default=DEFAULT_VISUAL_DIR)
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--object_name", type=str, default="3dprint")
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--sampling_method", type=str, default="euler")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--time_shifting_factor", type=float, default=3.0)
    parser.add_argument("--image_size", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--time_embed_layout", type=str, default="auto",
                        choices=["auto", "with_time_token", "without_time_token"])
    parser.add_argument("--vae_deterministic", action="store_true")
    parser.add_argument("--cast_model_bf16", action="store_true",
                        help="Debug only: reproduce the old inference/check behavior.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    weight_type = torch.bfloat16
    os.makedirs(args.output_dir, exist_ok=True)

    model, vae_model, text_tokenizer, showo_token_ids = load_models(
        args, device, weight_type
    )
    _, num_segment_tokens, max_text_len, add_time_token = resolve_video_token_layout(
        model, args.num_frames, args.time_embed_layout
    )
    args.num_segment_tokens = num_segment_tokens
    print(
        f"[4/4] Token layout: segment={num_segment_tokens}, "
        f"max_text_len={max_text_len}, add_time_token={add_time_token}"
    )

    dataset = build_training_dataset(
        args, text_tokenizer, showo_token_ids, num_segment_tokens
    )
    sample = find_object_sample(dataset, args.object_name)
    set_seed(args.seed)
    selected_indices = dataset._select_frame_indices(sample["frame_indices"])
    print(f"Selected {args.object_name} frame indices: {selected_indices}")

    visual_frames_a, loaded_indices_a = dataset.load_sample_video(
        sample, "visual", selected_indices
    )
    tactile_frames_a, loaded_tactile_indices = dataset.load_sample_video(
        sample, "tactile", selected_indices
    )
    if loaded_indices_a != loaded_tactile_indices:
        raise RuntimeError(
            f"Visual/tactile dataset frames differ: "
            f"visual={loaded_indices_a}, tactile={loaded_tactile_indices}"
        )
    visual_frames_b, loaded_indices_b = load_video_frames(
        args.visual_dir,
        args.num_frames,
        args.image_size,
        frame_indices=selected_indices,
        sampling_mode="contiguous",
        clip_start=0,
        return_indices=True,
    )
    if loaded_indices_a != loaded_indices_b:
        raise RuntimeError(
            f"Path A/B loaded different frames: A={loaded_indices_a}, B={loaded_indices_b}"
        )

    # Path A follows the training VAE layout: visual and tactile frames are encoded
    # together, then the first video segment is used as visual condition.
    set_seed(args.seed + 1)
    visual_latents_a, target_tactile_latents = encode_training_pair_latents(
        vae_model,
        visual_frames_a,
        tactile_frames_a,
        device,
        weight_type,
        args.vae_deterministic,
    )
    # Use the same VAE RNG state for Path B, so the visual condition matches
    # Path A even when stochastic VAE sampling is enabled.
    set_seed(args.seed + 1)
    visual_latents_b = encode_visual_latents(
        vae_model, visual_frames_b, device, weight_type, args.vae_deterministic
    )

    shared_z_tactile = torch.randn_like(visual_latents_a)

    token_batch_a = prepare_training_generation_batch(
        prompts=[args.text],
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        num_visual_tokens=num_segment_tokens,
        num_tactile_tokens=num_segment_tokens,
        max_text_len=max_text_len,
        device=device,
    )
    token_batch_b = prepare_tactile_gen_input_like_training(
        prompts=[args.text],
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        num_visual_tokens=num_segment_tokens,
        num_tactile_tokens=num_segment_tokens,
        max_text_len=max_text_len,
        device=device,
    )

    generated_a, trace_a = run_generation_path(
        "path_a_training_validation_sim",
        model,
        vae_model,
        visual_frames_a,
        visual_latents_a,
        shared_z_tactile,
        token_batch_a,
        selected_indices,
        args,
        device,
        weight_type,
        target_tactile_frames=tactile_frames_a,
        target_tactile_latents=target_tactile_latents,
    )
    generated_b, trace_b = run_generation_path(
        "path_b_inference_sim",
        model,
        vae_model,
        visual_frames_b,
        visual_latents_b,
        shared_z_tactile,
        token_batch_b,
        selected_indices,
        args,
        device,
        weight_type,
        target_tactile_frames=tactile_frames_a,
        target_tactile_latents=target_tactile_latents,
    )

    save_video_frames(
        visual_frames_a,
        os.path.join(args.output_dir, "visual_condition.mp4"),
        fps=2,
    )
    save_video_frames(
        tactile_frames_a,
        os.path.join(args.output_dir, "target_tactile.mp4"),
        fps=2,
    )
    save_video_frames(
        generated_a,
        os.path.join(args.output_dir, "path_a_generated.mp4"),
        fps=2,
    )
    save_video_frames(
        generated_b,
        os.path.join(args.output_dir, "path_b_generated.mp4"),
        fps=2,
    )
    write_trace(
        os.path.join(args.output_dir, "path_a_trace.txt"),
        "path_a_training_validation_sim",
        trace_a,
    )
    write_trace(
        os.path.join(args.output_dir, "path_b_trace.txt"),
        "path_b_inference_sim",
        trace_b,
    )
    write_compare_report(
        os.path.join(args.output_dir, "compare_report.txt"),
        trace_a,
        trace_b,
    )
    print(f"Done. Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
