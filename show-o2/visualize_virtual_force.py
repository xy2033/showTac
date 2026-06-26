#!/usr/bin/env python3
"""Visualize dense virtual-force labels used by tactile stage-one training."""

import argparse
import csv
import html
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-showo-force")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from datasets.tactile_visual_dataset import TactileVisualDataset


DEFAULT_DATA_ROOT = "/media/xy/Elements/tac/tacquad_gelsight_img/data_indoor"
DEFAULT_CSV_PATH = "/media/xy/Elements/tac/tacquad_gelsight_img/contact_indoor_list_tvl.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "virtual_force_viz"


class DummyTokenizer:
    pad_token_id = 0

    def __call__(
            self,
            text: str,
            add_special_tokens: bool = False,
            truncation: bool = True,
            max_length: int = 8192,
    ) -> Any:
        del text, add_special_tokens, truncation
        return type("TokenResult", (), {"input_ids": [1, 2, 3][:max_length]})()


SHOWO_TOKEN_IDS = {
    "bos_id": 1,
    "eos_id": 2,
    "bov_id": 3,
    "eov_id": 4,
    "vid_pad_id": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize virtual-force dense labels and contact masks.",
    )
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--csv_path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--num_visualize", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--frame_split_mode", choices=["contact_90_10", "legacy"], default="contact_90_10")
    parser.add_argument("--object_name", default=None)
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--scan_all", action="store_true")
    parser.add_argument("--image_size", type=int, default=432)
    parser.add_argument("--latent_height", type=int, default=27)
    parser.add_argument("--latent_width", type=int, default=27)
    parser.add_argument("--force_contact_threshold_percentile", type=float, default=75.0)
    parser.add_argument("--force_contact_abs_threshold", type=float, default=0.02)
    parser.add_argument("--force_min_contact_tokens", type=int, default=8)
    parser.add_argument("--force_contact_mask_soft", action="store_true")
    parser.add_argument("--force_contact_temporal_threshold", type=float, default=0.5)
    parser.add_argument("--force_dense_target_scale", type=float, default=5.0)
    parser.add_argument("--force_dense_target_clip", type=float, default=3.0)
    parser.add_argument(
        "--force_dense_reference_mode",
        choices=["background", "hybrid_temporal"],
        default="hybrid_temporal",
    )
    parser.add_argument("--mask_small_threshold", type=float, default=0.02)
    parser.add_argument("--clip_high_threshold", type=float, default=0.01)
    parser.add_argument("--near_zero_threshold", type=float, default=1e-5)
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> TactileVisualDataset:
    return TactileVisualDataset(
        data_root=args.data_root,
        csv_path=args.csv_path,
        text_tokenizer=DummyTokenizer(),
        max_seq_len=8192,
        image_size=args.image_size,
        latent_height=args.latent_height,
        latent_width=args.latent_width,
        num_frames=args.num_frames,
        num_visual_tokens_per_frame=args.latent_height * args.latent_width,
        num_tactile_tokens_per_frame=args.latent_height * args.latent_width,
        num_visual_tokens=args.num_frames * args.latent_height * args.latent_width,
        num_tactile_tokens=args.num_frames * args.latent_height * args.latent_width,
        cond_dropout_prob=0.0,
        split=args.split,
        frame_split_mode=args.frame_split_mode,
        showo_token_ids=SHOWO_TOKEN_IDS,
        compute_physics_targets=True,
        force_proxy_type="dense",
        force_contact_threshold_mode="percentile",
        force_contact_threshold_percentile=args.force_contact_threshold_percentile,
        force_contact_abs_threshold=args.force_contact_abs_threshold,
        force_min_contact_tokens=args.force_min_contact_tokens,
        force_contact_mask_soft=args.force_contact_mask_soft,
        force_dense_target_scale=args.force_dense_target_scale,
        force_dense_target_clip=args.force_dense_target_clip,
        force_dense_reference_mode=args.force_dense_reference_mode,
    )


def wan21_latent_frames(num_frames: int) -> int:
    if num_frames <= 1:
        return 1
    return 1 + math.ceil((num_frames - 1) / 4)


def compress_wan21_temporal_values_torch(values: torch.Tensor, latent_frames: int) -> torch.Tensor:
    """Match train_tactile_stage_one.compress_wan21_temporal_values for one clip."""
    if values.shape[0] == latent_frames:
        return values
    chunks = [values[:1]]
    cursor = 1
    while len(chunks) < latent_frames:
        if cursor >= values.shape[0]:
            chunks.append(values[-1:].clone())
        else:
            chunks.append(values[cursor:cursor + 4].mean(dim=0, keepdim=True))
            cursor += 4
    return torch.cat(chunks, dim=0)


def compress_wan21_temporal_values_np(values: np.ndarray, latent_frames: int) -> np.ndarray:
    return compress_wan21_temporal_values_torch(torch.as_tensor(values), latent_frames).cpu().numpy()


def select_dataset_indices(
        dataset: TactileVisualDataset,
        args: argparse.Namespace,
        rng: random.Random,
) -> List[int]:
    if args.object_name:
        matches = [
            idx for idx, sample in enumerate(dataset.samples)
            if sample["object_name"] == args.object_name
        ]
        if not matches:
            raise ValueError(f"Object not found in dataset split: {args.object_name}")
        return matches

    all_indices = list(range(len(dataset.samples)))
    if args.scan_all:
        return all_indices
    count = min(args.num_samples, len(all_indices))
    return rng.sample(all_indices, count)


def select_clip_frames(
        dataset: TactileVisualDataset,
        sample: Dict[str, Any],
        args: argparse.Namespace,
        rng: random.Random,
) -> Tuple[List[Image.Image], List[int]]:
    if args.start_frame is not None:
        requested = list(range(args.start_frame, args.start_frame + args.num_frames))
        return dataset._load_video_frames(sample["tactile_dir"], requested, return_indices=True)

    if dataset.frame_split_mode == "contact_90_10" and dataset.split == "train":
        ordered = sorted(sample["frame_indices"])
        if len(ordered) >= args.num_frames:
            max_start = len(ordered) - args.num_frames
            start = rng.randint(0, max_start) if max_start > 0 else 0
            selected = ordered[start:start + args.num_frames]
            return dataset._load_video_frames(sample["tactile_dir"], selected, return_indices=True)

    selected = dataset._select_frame_indices(sample["frame_indices"])
    return dataset._load_video_frames(sample["tactile_dir"], selected, return_indices=True)


def to_grid(values: np.ndarray, h: int, w: int) -> np.ndarray:
    return values.reshape(h, w)


def finite_stats(values: np.ndarray) -> Dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
    }


def summarize_clip(
        sample: Dict[str, Any],
        frame_indices: Sequence[int],
        force: np.ndarray,
        mask: np.ndarray,
        gate: np.ndarray,
        force_latent: np.ndarray,
        mask_latent: np.ndarray,
        args: argparse.Namespace,
) -> Dict[str, Any]:
    force_stats = finite_stats(force)
    force_norm = np.linalg.norm(force, axis=-1)
    latent_norm = np.linalg.norm(force_latent, axis=-1)
    mask_ratio = mask.mean(axis=(1, 2))
    latent_mask_ratio = mask_latent.mean(axis=(1, 2))
    clip_hit_ratio = 0.0
    if args.force_dense_target_clip > 0.0:
        clip_hit_ratio = float((np.abs(force) >= args.force_dense_target_clip - 1e-6).mean())

    has_nan = bool(np.isnan(force).any() or np.isnan(mask).any())
    has_inf = bool(np.isinf(force).any() or np.isinf(mask).any())
    flags = []
    if has_nan or has_inf:
        flags.append("nan_or_inf")
    if float(mask.mean()) <= 0.0:
        flags.append("empty_mask")
    if float(mask.mean()) < args.mask_small_threshold:
        flags.append("small_mask")
    if float(mask_latent.mean()) <= 0.0:
        flags.append("latent_mask_empty")
    if float(clip_hit_ratio) > args.clip_high_threshold:
        flags.append("clip_high")
    if float(force_norm.mean()) < args.near_zero_threshold:
        flags.append("force_near_zero")

    row = {
        "object_name": sample["object_name"],
        "frame_indices": " ".join(str(idx) for idx in frame_indices),
        "num_frames": int(force.shape[0]),
        "latent_frames": int(force_latent.shape[0]),
        "force_shape": str(tuple(force.shape)),
        "mask_shape": str(tuple(mask.shape)),
        "latent_force_shape": str(tuple(force_latent.shape)),
        "latent_mask_shape": str(tuple(mask_latent.shape)),
        "force_min": force_stats["min"],
        "force_max": force_stats["max"],
        "force_mean": force_stats["mean"],
        "force_std": force_stats["std"],
        "force_norm_mean": float(force_norm.mean()),
        "force_norm_p95": float(np.percentile(force_norm, 95)),
        "latent_force_norm_mean": float(latent_norm.mean()),
        "mask_ratio_mean": float(mask.mean()),
        "mask_ratio_per_frame": " ".join(f"{x:.6f}" for x in mask_ratio),
        "latent_mask_ratio_mean": float(mask_latent.mean()),
        "latent_mask_ratio_per_frame": " ".join(f"{x:.6f}" for x in latent_mask_ratio),
        "clip_hit_ratio": clip_hit_ratio,
        "contact_gate": " ".join(f"{float(x):.6f}" for x in gate),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "flags": ",".join(flags) if flags else "ok",
    }
    return row


def aligned_tactile_arrays(
        dataset: TactileVisualDataset,
        sample: Dict[str, Any],
        frames: Sequence[Image.Image],
) -> Tuple[np.ndarray, np.ndarray]:
    ref = dataset._geometric_align(dataset._load_no_contact_reference(sample["tactile_dir"]))
    rgb = np.stack([dataset._geometric_align(frame) for frame in frames], axis=0)
    return ref, rgb


def plot_frame_row(
        axes: Sequence[plt.Axes],
        rgb: np.ndarray,
        ref: np.ndarray,
        force_tokens: np.ndarray,
        mask_tokens: np.ndarray,
        title_prefix: str,
        h: int,
        w: int,
        norm_vmax: float,
        div_absmax: float,
) -> None:
    diff = np.mean(np.abs(rgb - ref), axis=-1) / 255.0
    ux = to_grid(force_tokens[:, 0], h, w)
    uy = to_grid(force_tokens[:, 1], h, w)
    div = to_grid(force_tokens[:, 2], h, w)
    force_norm = np.sqrt(ux * ux + uy * uy + div * div)
    mask = to_grid(mask_tokens[:, 0], h, w)

    axes[0].imshow(np.clip(rgb / 255.0, 0.0, 1.0))
    axes[0].set_title(f"{title_prefix} tactile")
    axes[1].imshow(diff, cmap="magma")
    axes[1].set_title("background diff")
    axes[2].imshow(np.clip(rgb / 255.0, 0.0, 1.0))
    axes[2].imshow(mask, cmap="Greens", alpha=0.45, vmin=0.0, vmax=max(1.0, float(mask.max())))
    axes[2].set_title(f"mask ratio {mask.mean():.3f}")
    im_norm = axes[3].imshow(force_norm, cmap="viridis", vmin=0.0, vmax=norm_vmax)
    axes[3].set_title("force norm")
    step = max(1, h // 9)
    yy, xx = np.mgrid[0:h:step, 0:w:step]
    axes[4].imshow(force_norm, cmap="Greys", alpha=0.35, vmin=0.0, vmax=norm_vmax)
    axes[4].quiver(xx, yy, ux[::step, ::step], -uy[::step, ::step], color="tab:red", angles="xy")
    axes[4].set_title("ux/uy quiver")
    im_div = axes[5].imshow(div, cmap="coolwarm", vmin=-div_absmax, vmax=div_absmax)
    axes[5].set_title("div_u")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    return im_norm, im_div


def make_panel(
        out_path: Path,
        sample: Dict[str, Any],
        frame_indices: Sequence[int],
        ref_rgb: np.ndarray,
        rgb_frames: np.ndarray,
        force: np.ndarray,
        mask: np.ndarray,
        force_latent: np.ndarray,
        mask_latent: np.ndarray,
        summary: Dict[str, Any],
        args: argparse.Namespace,
) -> None:
    h = args.latent_height
    w = args.latent_width
    rows = force.shape[0] + force_latent.shape[0]
    cols = 6
    norm_vmax = float(np.percentile(np.linalg.norm(force, axis=-1), 99))
    norm_vmax = max(norm_vmax, 1e-6)
    div_absmax = float(np.percentile(np.abs(force[..., 2]), 99))
    div_absmax = max(div_absmax, 1e-6)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.7), squeeze=False)
    fig.suptitle(
        f"{sample['object_name']} frames={list(frame_indices)} "
        f"flags={summary['flags']} force=[{summary['force_min']:.3g},{summary['force_max']:.3g}] "
        f"std={summary['force_std']:.3g} clip={summary['clip_hit_ratio']:.4f}",
        fontsize=13,
    )

    norm_image = None
    div_image = None
    for i in range(force.shape[0]):
        norm_image, div_image = plot_frame_row(
            axes[i],
            rgb_frames[i],
            ref_rgb,
            force[i],
            mask[i],
            f"raw f{i}",
            h,
            w,
            norm_vmax,
            div_absmax,
        )

    for j in range(force_latent.shape[0]):
        row = force.shape[0] + j
        src_idx = 0 if j == 0 else min(1 + (j - 1) * 4, rgb_frames.shape[0] - 1)
        norm_image, div_image = plot_frame_row(
            axes[row],
            rgb_frames[src_idx],
            ref_rgb,
            force_latent[j],
            mask_latent[j],
            f"latent f{j}",
            h,
            w,
            norm_vmax,
            div_absmax,
        )

    if norm_image is not None:
        fig.colorbar(norm_image, ax=axes[:, 3], fraction=0.015, pad=0.01)
    if div_image is not None:
        fig.colorbar(div_image, ax=axes[:, 5], fraction=0.015, pad=0.01)
    fig.subplots_adjust(top=0.93, wspace=0.08, hspace=0.35, right=0.96)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_index(path: Path, rows: Sequence[Dict[str, Any]], image_names: Sequence[str]) -> None:
    cards = []
    image_by_key = {
        Path(name).stem: name for name in image_names
    }
    for row in rows:
        key = row.get("panel_key")
        image_name = image_by_key.get(key)
        flags = html.escape(str(row["flags"]))
        table = "".join(
            f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
            for k, v in row.items()
            if k != "panel_key"
        )
        image_html = ""
        if image_name:
            image_html = f'<a href="{html.escape(image_name)}"><img src="{html.escape(image_name)}" /></a>'
        cards.append(
            f"<section class='card {flags}'><h2>{html.escape(row['object_name'])} "
            f"<span>{flags}</span></h2>{image_html}<table>{table}</table></section>"
        )
    content = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Virtual Force Label Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2937; }}
    h1 {{ margin-bottom: 8px; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 16px; margin: 18px 0; }}
    .card h2 {{ margin: 0 0 12px; font-size: 20px; }}
    .card h2 span {{ font-size: 13px; font-weight: 600; color: #b45309; margin-left: 8px; }}
    img {{ width: 100%; height: auto; border: 1px solid #e5e7eb; }}
    table {{ border-collapse: collapse; margin-top: 12px; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ width: 230px; color: #4b5563; }}
  </style>
</head>
<body>
  <h1>Virtual Force Label Report</h1>
  <p>Dense labels and contact masks are computed through <code>TactileVisualDataset</code>; latent rows use Wan2.1 temporal compression.</p>
  {''.join(cards)}
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def process_one(
        dataset: TactileVisualDataset,
        sample_idx: int,
        args: argparse.Namespace,
        rng: random.Random,
        panel_dir: Path,
        make_visual: bool,
) -> Tuple[Dict[str, Any], Optional[str]]:
    sample = dataset.samples[sample_idx]
    frames, frame_indices = select_clip_frames(dataset, sample, args, rng)
    force_t, mask_t, gate_t = dataset._compute_dense_force_targets(sample["tactile_dir"], frames)
    force = force_t.cpu().numpy()
    mask = mask_t.cpu().numpy()
    gate = gate_t.cpu().numpy()
    latent_frames = wan21_latent_frames(args.num_frames)
    force_latent = compress_wan21_temporal_values_np(force, latent_frames)
    mask_latent = compress_wan21_temporal_values_np(mask, latent_frames)
    mask_latent = (mask_latent >= args.force_contact_temporal_threshold).astype(np.float32)
    summary = summarize_clip(sample, frame_indices, force, mask, gate, force_latent, mask_latent, args)
    panel_key = f"{sample_idx:04d}_{sample['object_name']}_{'_'.join(str(i) for i in frame_indices)}"
    safe_key = "".join(c if c.isalnum() or c in "._-" else "_" for c in panel_key)
    summary["panel_key"] = safe_key

    image_name = None
    if make_visual:
        ref_rgb, rgb_frames = aligned_tactile_arrays(dataset, sample, frames)
        image_name = f"{safe_key}.png"
        make_panel(
            panel_dir / image_name,
            sample,
            frame_indices,
            ref_rgb,
            rgb_frames,
            force,
            mask,
            force_latent,
            mask_latent,
            summary,
            args,
        )
    return summary, image_name


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir).resolve()
    panel_dir = out_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)
    if len(dataset.samples) == 0:
        raise RuntimeError("No usable tactile samples were loaded.")

    sample_indices = select_dataset_indices(dataset, args, rng)
    if args.object_name and args.start_frame is not None:
        sample_indices = sample_indices[:1]

    visualize_limit = args.num_visualize
    if visualize_limit is None:
        visualize_limit = args.num_samples if not args.scan_all else min(args.num_samples, len(sample_indices))
    visualize_set = set(sample_indices[:max(0, visualize_limit)])

    rows = []
    image_names = []
    for position, sample_idx in enumerate(sample_indices, start=1):
        make_visual = sample_idx in visualize_set
        row, image_name = process_one(dataset, sample_idx, args, rng, panel_dir, make_visual)
        rows.append(row)
        if image_name:
            image_names.append(f"panels/{image_name}")
        print(
            f"[{position}/{len(sample_indices)}] {row['object_name']} "
            f"frames={row['frame_indices']} flags={row['flags']} "
            f"mask={row['mask_ratio_mean']:.4f} clip={row['clip_hit_ratio']:.4f}"
        )

    write_summary_csv(out_dir / "summary.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_index(out_dir / "index.html", rows, image_names)

    flag_counts: Dict[str, int] = {}
    for row in rows:
        for flag in str(row["flags"]).split(","):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    print(f"Wrote report: {out_dir / 'index.html'}")
    print(f"Wrote summary: {out_dir / 'summary.csv'}")
    print(f"Flag counts: {flag_counts}")


if __name__ == "__main__":
    main()
