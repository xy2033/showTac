#!/usr/bin/env python3
"""Diagnose 2-frame long-baseline RAFT flow on existing tactile debug windows."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from precompute_tactile_raft_flow import (
    compute_flow,
    load_raft,
    load_tactile_frame,
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


def flow_to_rgb_fixed(flow: np.ndarray, max_mag: float = 4.0, gamma: float = 0.5, min_mag: float = 0.05) -> np.ndarray:
    flow = np.nan_to_num(flow.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    dx = flow[..., 0]
    dy = flow[..., 1]
    mag = np.sqrt(dx * dx + dy * dy)
    value = np.clip(mag / max(max_mag, 1e-6), 0.0, 1.0) ** gamma
    value[mag < min_mag] = 0.0

    hue = (np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi)
    h6 = hue * 6.0
    sector = np.floor(h6).astype(np.int32) % 6
    frac = h6 - np.floor(h6)
    saturation = np.ones_like(value)
    p = value * (1.0 - saturation)
    q = value * (1.0 - frac * saturation)
    t = value * (1.0 - (1.0 - frac) * saturation)

    rgb = np.zeros((*flow.shape[:2], 3), dtype=np.float32)
    choices = (
        np.stack([value, t, p], axis=-1),
        np.stack([q, value, p], axis=-1),
        np.stack([p, value, t], axis=-1),
        np.stack([p, q, value], axis=-1),
        np.stack([t, p, value], axis=-1),
        np.stack([value, p, q], axis=-1),
    )
    for idx, choice in enumerate(choices):
        rgb[sector == idx] = choice[sector == idx]
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def contact_mask(frame: Image.Image, background: Image.Image, threshold: float) -> np.ndarray:
    diff = np.mean(
        np.abs(np.asarray(frame, dtype=np.float32) - np.asarray(background, dtype=np.float32)),
        axis=-1,
    )
    return (diff > threshold).astype(np.uint8) * 255


def find_windows(debug_input_dir: Path, objects: list[str]) -> list[Path]:
    windows = []
    for object_name in objects:
        object_dir = debug_input_dir / object_name
        if not object_dir.is_dir():
            print(f"Warning: missing debug object dir: {object_dir}")
            continue
        windows.extend(sorted(path for path in object_dir.iterdir() if path.is_dir()))
    return windows


def save_compare_grid(
    path: Path,
    tactile_frames: list[Image.Image],
    masks: list[np.ndarray],
    adjacent_flows: list[np.ndarray],
    long_flows_expanded: list[np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    rows = [
        ("tactile", tactile_frames),
        ("contact mask", masks),
        ("adjacent RAFT", adjacent_flows),
        ("long RAFT", long_flows_expanded),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(rows), len(tactile_frames), figsize=(2.2 * len(tactile_frames), 8.8))
    for row_idx, (label, images) in enumerate(rows):
        for col_idx, image in enumerate(images):
            if row_idx == 1:
                axes[row_idx, col_idx].imshow(image, cmap="gray", vmin=0, vmax=255)
            else:
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
    if len(selected_indices) != 5:
        raise ValueError(f"Expected 5 selected indices in {window_dir}, got {selected_indices}")

    tactile_dir = args.data_root / object_name / "gelsight"
    tactile_frames = [load_tactile_frame(tactile_dir, idx, args.resolution) for idx in selected_indices]
    background = load_tactile_frame(tactile_dir, 0, args.resolution)
    masks = [contact_mask(frame, background, args.contact_threshold) for frame in tactile_frames]

    long_flow_00_to_02 = compute_flow(model, raft_transforms, tactile_frames[0], tactile_frames[2], device)
    long_flow_02_to_04 = compute_flow(model, raft_transforms, tactile_frames[2], tactile_frames[4], device)
    adjacent_flows_raw = [
        compute_flow(model, raft_transforms, tactile_frames[idx], tactile_frames[idx + 1], device)
        for idx in range(4)
    ]
    long_flow_00_to_02_rgb = flow_to_rgb_fixed(long_flow_00_to_02, args.fixed_max_mag, args.gamma, args.min_mag)
    long_flow_02_to_04_rgb = flow_to_rgb_fixed(long_flow_02_to_04, args.fixed_max_mag, args.gamma, args.min_mag)
    adjacent_flows = [
        flow_to_rgb_fixed(flow, args.fixed_max_mag, args.gamma, args.min_mag)
        for flow in adjacent_flows_raw
    ]
    adjacent_flows.append(adjacent_flows[-1])

    long_flows_expanded = [
        long_flow_00_to_02_rgb,
        long_flow_00_to_02_rgb,
        long_flow_02_to_04_rgb,
        long_flow_02_to_04_rgb,
        long_flow_02_to_04_rgb,
    ]

    out_dir = args.output_dir / object_name / window_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(tactile_frames):
        frame.save(out_dir / f"tactile_{idx:02d}.png")
    for idx, mask in enumerate(masks):
        save_png(out_dir / f"contact_mask_{idx:02d}.png", mask)
    np.save(out_dir / "long_flow_00_to_02.npy", long_flow_00_to_02)
    np.save(out_dir / "long_flow_02_to_04.npy", long_flow_02_to_04)
    for idx, flow in enumerate(adjacent_flows_raw):
        np.save(out_dir / f"adjacent_flow_{idx:02d}_to_{idx + 1:02d}.npy", flow)
        save_png(out_dir / f"adjacent_flow_{idx:02d}_to_{idx + 1:02d}_fixed_rgb.png", adjacent_flows[idx])
    save_png(out_dir / "long_flow_00_to_02_rgb.png", long_flow_00_to_02_rgb)
    save_png(out_dir / "long_flow_02_to_04_rgb.png", long_flow_02_to_04_rgb)
    save_compare_grid(out_dir / "long_baseline_compare_grid.png", tactile_frames, masks, adjacent_flows, long_flows_expanded)
    video_backend = save_flow_video(out_dir / "long_flow_video.mp4", long_flows_expanded)

    save_json(
        out_dir / "metadata.json",
        {
            "object_name": object_name,
            "source_window_dir": str(window_dir),
            "selected_indices": selected_indices,
            "long_flow_mapping": {
                "long_flow_00_to_02": "frame_0_to_frame_2",
                "long_flow_02_to_04": "frame_2_to_frame_4",
            },
            "display_expansion_note": "5-column visualization expands two long flows as [F00_to_02, F00_to_02, F02_to_04, F02_to_04, F02_to_04]; it is not five independent targets.",
            "raft_input": "original_tactile_rgb",
            "adjacent_flow_row": "recomputed adjacent RAFT flows rendered with the same fixed_scale_gamma settings; 5th column repeats frame_3_to_frame_4.",
            "flow_visualization": {
                "mode": "fixed_scale_gamma",
                "fixed_max_mag": args.fixed_max_mag,
                "gamma": args.gamma,
                "min_mag": args.min_mag,
            },
            "contact_mask": {
                "background_frame": str(tactile_dir / "0.png"),
                "threshold": args.contact_threshold,
                "formula": "mean(abs(frame - background), channel) > threshold",
                "used_for_raft": False,
            },
            "video_backend": video_backend,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path("/media/xy/Elements/tac/tacquad_gelsight_img/data_indoor"))
    parser.add_argument("--debug_input_dir", type=Path, default=Path("/media/xy/Elements/showO/outputs/flow_debug"))
    parser.add_argument("--output_dir", type=Path, default=Path("/media/xy/Elements/showO/outputs/flow_debug_long_baseline"))
    parser.add_argument("--objects", nargs="*", default=DEFAULT_OBJECTS)
    parser.add_argument("--resolution", type=int, default=432)
    parser.add_argument("--fixed_max_mag", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--min_mag", type=float, default=0.05)
    parser.add_argument("--contact_threshold", type=float, default=8.0)
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
