#!/usr/bin/env python3
"""Run RAFT on background-subtracted tactile debug windows."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from precompute_tactile_raft_flow import (
    compute_flow,
    flow_to_rgb,
    load_raft,
    load_tactile_frame,
    pad_flow_sequence,
    save_flow_video,
    save_json,
    save_png,
)


DEFAULT_OBJECTS = [
    "can_metal_hard_smooth_top",
    "glass_jar",
    "peach",
    "plastic_yellow_duck_beak",
    "sponge_brush_handle",
]


def bgsub_frame(frame: Image.Image, background: Image.Image, gain: float, threshold: float) -> Image.Image:
    frame_arr = np.asarray(frame, dtype=np.float32)
    bg_arr = np.asarray(background, dtype=np.float32)
    residual = frame_arr - bg_arr
    residual[np.abs(residual) < threshold] = 0.0
    # Neutral gray means "same as background"; positive/negative residuals deviate from gray.
    out = np.clip(127.5 + gain * residual, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def find_windows(debug_input_dir: Path, objects: list[str]) -> list[Path]:
    windows = []
    for object_name in objects:
        object_dir = debug_input_dir / object_name
        if not object_dir.is_dir():
            print(f"Warning: missing debug object dir: {object_dir}")
            continue
        windows.extend(sorted(path for path in object_dir.iterdir() if path.is_dir()))
    return windows


def load_old_flows(window_dir: Path) -> list[np.ndarray]:
    flows = []
    for idx in range(4):
        flows.append(np.asarray(Image.open(window_dir / f"flow_{idx:02d}_to_{idx + 1:02d}_rgb.png").convert("RGB")))
    flows.append(np.asarray(Image.open(window_dir / "flow_04_pad_repeat_last_rgb.png").convert("RGB")))
    return flows


def save_compare_grid(
    path: Path,
    tactile_frames: list[Image.Image],
    bgsub_frames: list[Image.Image],
    old_flows: list[np.ndarray],
    bgsub_flows: list[np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    rows = [
        ("tactile", tactile_frames),
        ("frame - 0.png", bgsub_frames),
        ("RAFT original", old_flows),
        ("RAFT bgsub", bgsub_flows),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(rows), len(tactile_frames), figsize=(2.2 * len(tactile_frames), 8.8))
    for row_idx, (label, images) in enumerate(rows):
        for col_idx, image in enumerate(images):
            axes[row_idx, col_idx].imshow(image)
            axes[row_idx, col_idx].axis("off")
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(label)
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(f"frame {col_idx}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.inference_mode()
def process_window(window_dir: Path, args: argparse.Namespace, model, raft_transforms, device: torch.device) -> None:
    metadata = json.loads((window_dir / "metadata.json").read_text())
    object_name = metadata["object_name"]
    selected_indices = metadata["selected_indices"]
    tactile_dir = args.data_root / object_name / "gelsight"
    background = load_tactile_frame(tactile_dir, 0, args.resolution)

    tactile_frames = [load_tactile_frame(tactile_dir, idx, args.resolution) for idx in selected_indices]
    bgsub_frames = [
        bgsub_frame(frame, background, gain=args.bg_gain, threshold=args.bg_threshold)
        for frame in tactile_frames
    ]

    flows = [
        compute_flow(model, raft_transforms, bgsub_frames[idx], bgsub_frames[idx + 1], device)
        for idx in range(len(bgsub_frames) - 1)
    ]
    max_mag = max(float(np.percentile(np.concatenate([np.linalg.norm(flow, axis=-1).reshape(-1) for flow in flows]), 95)), 1e-6)
    bgsub_flow_rgbs = [flow_to_rgb(flow, max_mag=max_mag) for flow in flows]
    bgsub_flow_rgbs = pad_flow_sequence(bgsub_flow_rgbs, "repeat_last")
    old_flows = load_old_flows(window_dir)

    out_dir = args.output_dir / object_name / window_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(tactile_frames):
        frame.save(out_dir / f"tactile_{idx:02d}.png")
    for idx, frame in enumerate(bgsub_frames):
        frame.save(out_dir / f"bgsub_{idx:02d}.png")
    for idx, flow_rgb in enumerate(bgsub_flow_rgbs[:-1]):
        save_png(out_dir / f"bgsub_flow_{idx:02d}_to_{idx + 1:02d}_rgb.png", flow_rgb)
    save_png(out_dir / "bgsub_flow_04_pad_repeat_last_rgb.png", bgsub_flow_rgbs[-1])
    save_compare_grid(out_dir / "bgsub_compare_grid.png", tactile_frames, bgsub_frames, old_flows, bgsub_flow_rgbs)
    video_backend = save_flow_video(out_dir / "bgsub_flow_video.mp4", bgsub_flow_rgbs)

    save_json(
        out_dir / "metadata.json",
        {
            "object_name": object_name,
            "source_window_dir": str(window_dir),
            "selected_indices": selected_indices,
            "background_frame": str(tactile_dir / "0.png"),
            "bgsub_encoding": "clip(127.5 + bg_gain * ((frame - background) with abs<threshold set to 0), 0, 255)",
            "bg_gain": args.bg_gain,
            "bg_threshold": args.bg_threshold,
            "normalization": "window_level_p95",
            "max_mag": max_mag,
            "padding_mode": "repeat_last",
            "video_backend": video_backend,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--debug_input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--objects", nargs="*", default=DEFAULT_OBJECTS)
    parser.add_argument("--resolution", type=int, default=432)
    parser.add_argument("--bg_gain", type=float, default=4.0)
    parser.add_argument("--bg_threshold", type=float, default=2.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = find_windows(args.debug_input_dir, args.objects)
    if not windows:
        raise RuntimeError("No debug windows found.")
    device = torch.device(args.device)
    model, raft_transforms, weights_name = load_raft(device)
    print(f"RAFT weights: {weights_name}; device: {device}")
    print(f"Processing {len(windows)} windows -> {args.output_dir}")
    for idx, window_dir in enumerate(windows, start=1):
        print(f"[{idx}/{len(windows)}] {window_dir}")
        process_window(window_dir, args, model, raft_transforms, device)
    print("Done.")


if __name__ == "__main__":
    main()
