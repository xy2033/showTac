#!/usr/bin/env python3
"""Diagnose contact-masked background-difference motion targets."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from precompute_tactile_raft_flow import (
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


def find_windows(debug_input_dir: Path, objects: list[str]) -> list[Path]:
    windows = []
    for object_name in objects:
        object_dir = debug_input_dir / object_name
        if not object_dir.is_dir():
            print(f"Warning: missing debug object dir: {object_dir}")
            continue
        windows.extend(sorted(path for path in object_dir.iterdir() if path.is_dir()))
    return windows


def dilate_mask(mask: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return mask
    if size % 2 == 0:
        raise ValueError(f"mask_dilate must be odd, got {size}")
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    return np.asarray(image.filter(ImageFilter.MaxFilter(size)), dtype=np.uint8) > 0


def residual_and_mask(
    frame: Image.Image,
    background: Image.Image,
    threshold: float,
    mask_dilate: int,
) -> tuple[np.ndarray, np.ndarray]:
    residual = np.asarray(frame, dtype=np.float32) - np.asarray(background, dtype=np.float32)
    contact = np.mean(np.abs(residual), axis=-1) > threshold
    contact = dilate_mask(contact, mask_dilate)
    return residual, contact


def encode_motion(motion: np.ndarray, mask: np.ndarray, motion_max: float, gamma: float) -> np.ndarray:
    motion = motion * mask.astype(np.float32)
    value = np.clip(motion / max(motion_max, 1e-6), 0.0, 1.0) ** gamma
    gray = np.clip(value * 255.0, 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def diff_motion(
    residual_a: np.ndarray,
    residual_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    motion_max: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = mask_a | mask_b
    motion = np.mean(np.abs(residual_b - residual_a), axis=-1)
    return motion * mask.astype(np.float32), encode_motion(motion, mask, motion_max, gamma)


def save_compare_grid(
    path: Path,
    tactile_frames: list[Image.Image],
    masks: list[np.ndarray],
    adjacent_targets: list[np.ndarray],
    long_targets: list[np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    rows = [
        ("tactile", tactile_frames),
        ("contact mask", [mask.astype(np.uint8) * 255 for mask in masks]),
        ("adjacent diff", adjacent_targets),
        ("long diff", long_targets),
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


def process_window(window_dir: Path, args: argparse.Namespace) -> None:
    metadata = json.loads((window_dir / "metadata.json").read_text())
    object_name = metadata["object_name"]
    selected_indices = metadata["selected_indices"]
    if len(selected_indices) != 5:
        raise ValueError(f"Expected 5 selected indices in {window_dir}, got {selected_indices}")

    tactile_dir = args.data_root / object_name / "gelsight"
    background = load_tactile_frame(tactile_dir, 0, args.resolution)
    tactile_frames = [load_tactile_frame(tactile_dir, idx, args.resolution) for idx in selected_indices]
    residuals = []
    masks = []
    for frame in tactile_frames:
        residual, mask = residual_and_mask(frame, background, args.contact_threshold, args.mask_dilate)
        residuals.append(residual)
        masks.append(mask)

    adjacent_motion = []
    adjacent_targets = []
    for idx in range(4):
        motion, target = diff_motion(
            residuals[idx],
            residuals[idx + 1],
            masks[idx],
            masks[idx + 1],
            args.motion_max,
            args.motion_gamma,
        )
        adjacent_motion.append(motion)
        adjacent_targets.append(target)
    adjacent_targets.append(adjacent_targets[-1])

    long_motion_00_to_02, long_target_00_to_02 = diff_motion(
        residuals[0], residuals[2], masks[0], masks[2], args.motion_max, args.motion_gamma
    )
    long_motion_02_to_04, long_target_02_to_04 = diff_motion(
        residuals[2], residuals[4], masks[2], masks[4], args.motion_max, args.motion_gamma
    )
    long_targets = [
        long_target_00_to_02,
        long_target_00_to_02,
        long_target_02_to_04,
        long_target_02_to_04,
        long_target_02_to_04,
    ]

    out_dir = args.output_dir / object_name / window_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(tactile_frames):
        frame.save(out_dir / f"tactile_{idx:02d}.png")
    for idx, mask in enumerate(masks):
        save_png(out_dir / f"contact_mask_{idx:02d}.png", mask.astype(np.uint8) * 255)
    for idx, (motion, target) in enumerate(zip(adjacent_motion, adjacent_targets[:-1])):
        np.save(out_dir / f"adjacent_diff_motion_{idx:02d}_to_{idx + 1:02d}.npy", motion.astype(np.float32))
        save_png(out_dir / f"adjacent_diff_motion_{idx:02d}_to_{idx + 1:02d}_rgb.png", target)
    np.save(out_dir / "diff_motion_00_to_02.npy", long_motion_00_to_02.astype(np.float32))
    np.save(out_dir / "diff_motion_02_to_04.npy", long_motion_02_to_04.astype(np.float32))
    save_png(out_dir / "diff_motion_00_to_02_rgb.png", long_target_00_to_02)
    save_png(out_dir / "diff_motion_02_to_04_rgb.png", long_target_02_to_04)
    save_compare_grid(out_dir / "diff_motion_compare_grid.png", tactile_frames, masks, adjacent_targets, long_targets)
    video_backend = save_flow_video(out_dir / "diff_motion_video.mp4", long_targets)

    save_json(
        out_dir / "metadata.json",
        {
            "object_name": object_name,
            "source_window_dir": str(window_dir),
            "selected_indices": selected_indices,
            "background_frame": str(tactile_dir / "0.png"),
            "residual": "R_t = frame_t - background",
            "contact_mask": {
                "formula": "mean(abs(R_t), channel) > contact_threshold",
                "contact_threshold": args.contact_threshold,
                "mask_dilate": args.mask_dilate,
                "dilation": "PIL.ImageFilter.MaxFilter",
            },
            "long_diff_motion": {
                "diff_motion_00_to_02": "mean(abs(R_2 - R_0), channel) * (C_0 OR C_2)",
                "diff_motion_02_to_04": "mean(abs(R_4 - R_2), channel) * (C_2 OR C_4)",
            },
            "display_expansion_note": "5-column long diff row expands two targets as [M00_to_02, M00_to_02, M02_to_04, M02_to_04, M02_to_04]; it is not five independent targets.",
            "visualization": {
                "motion_max": args.motion_max,
                "motion_gamma": args.motion_gamma,
                "mask_outside_forced_black": True,
                "rgb_encoding": "grayscale repeated to 3 channels",
            },
            "video_backend": video_backend,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path("/media/xy/Elements/tac/tacquad_gelsight_img/data_indoor"))
    parser.add_argument("--debug_input_dir", type=Path, default=Path("/media/xy/Elements/showO/outputs/flow_debug"))
    parser.add_argument("--output_dir", type=Path, default=Path("/media/xy/Elements/showO/outputs/flow_debug_diff_motion"))
    parser.add_argument("--objects", nargs="*", default=DEFAULT_OBJECTS)
    parser.add_argument("--resolution", type=int, default=432)
    parser.add_argument("--contact_threshold", type=float, default=8.0)
    parser.add_argument("--mask_dilate", type=int, default=5)
    parser.add_argument("--motion_max", type=float, default=32.0)
    parser.add_argument("--motion_gamma", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = find_windows(args.debug_input_dir, args.objects)
    if not windows:
        raise RuntimeError("No debug windows found.")
    print(f"Processing {len(windows)} windows -> {args.output_dir}")
    for idx, window_dir in enumerate(windows, start=1):
        print(f"[{idx}/{len(windows)}] {window_dir}")
        process_window(window_dir, args)
    print("Done.")


if __name__ == "__main__":
    main()
