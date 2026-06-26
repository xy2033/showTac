#!/usr/bin/env python3
"""Precompute contact-aware diff-motion targets for tactile videos."""

import argparse
import ast
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.tactile_motion import (  # noqa: E402  shared core (dedup)
    compute_contact_masks,
    compute_motion_targets,
    contact_mask as motion_contact_mask,
    debaseline,
    median_filter_3x3,
    area_pool_2d,
)


CACHE_VERSION = 1
DEFAULT_DATA_ROOT = Path("/media/xy/Elements/tac/tacquad_gelsight_img/data_indoor")
DEFAULT_CSV_PATH = Path("/media/xy/Elements/tac/tacquad_gelsight_img/contact_indoor_list_tvl.csv")


try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:  # pragma: no cover - old Pillow compatibility
    RESAMPLE_BICUBIC = Image.BICUBIC
    RESAMPLE_NEAREST = Image.NEAREST


def parse_index_list(value):
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [int(idx) for idx in parsed]


def resolve_contact_split_indices(start_idx, end_idx, split):
    if end_idx < start_idx:
        return []
    indices = list(range(start_idx, end_idx + 1))
    if split == "all":
        return indices
    boundary = int(np.floor(len(indices) * 0.9))
    if len(indices) >= 2:
        boundary = min(max(boundary, 1), len(indices) - 1)
    if split == "train":
        return indices[:boundary]
    if split == "test":
        return indices[boundary:]
    raise ValueError(f"Unsupported split: {split}")


def resolve_legacy_split_indices(train_indices, test_indices, split):
    if split == "train":
        return train_indices
    if split == "test":
        return test_indices
    if split != "all":
        raise ValueError(f"Unsupported split: {split}")
    seen = set()
    output = []
    for idx in train_indices + test_indices:
        if idx not in seen:
            output.append(idx)
            seen.add(idx)
    return output


def list_frame_numbers(frame_dir):
    return sorted(
        int(path.stem)
        for path in Path(frame_dir).iterdir()
        if path.suffix == ".png" and path.stem.isdigit()
    )


def filter_paired_frame_indices(indices, visual_dir, tactile_dir):
    paired = set(list_frame_numbers(visual_dir)) & set(list_frame_numbers(tactile_dir))
    return [idx for idx in indices if idx in paired]


def load_samples(data_root, csv_path, split, frame_split_mode, objects=None):
    object_filter = set(objects or [])
    samples = []
    with Path(csv_path).open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 11:
                continue
            try:
                object_name = row[0].strip()
                contact_start_idx = int(row[1].strip())
                contact_end_idx = int(row[2].strip())
                train_indices = parse_index_list(row[-2].strip())
                test_indices = parse_index_list(row[-1].strip())
            except (IndexError, ValueError):
                continue
            if object_filter and object_name not in object_filter:
                continue

            object_dir = Path(data_root) / object_name
            visual_dir = object_dir / "img_gelsight"
            tactile_dir = object_dir / "gelsight"
            if not visual_dir.is_dir() or not tactile_dir.is_dir():
                print(f"Warning: skipping {object_name} - missing img_gelsight/ or gelsight/")
                continue

            if frame_split_mode == "contact_90_10":
                frame_indices = resolve_contact_split_indices(contact_start_idx, contact_end_idx, split)
            else:
                frame_indices = resolve_legacy_split_indices(train_indices, test_indices, split)
            frame_indices = filter_paired_frame_indices(frame_indices, visual_dir, tactile_dir)
            if len(frame_indices) < 3:
                print(f"Warning: skipping {object_name} - fewer than 3 paired frames")
                continue

            samples.append(
                {
                    "object_name": object_name,
                    "visual_dir": visual_dir,
                    "tactile_dir": tactile_dir,
                    "frame_indices": sorted(frame_indices),
                    "contact_start_idx": contact_start_idx,
                    "contact_end_idx": contact_end_idx,
                    "split": split,
                }
            )
    return samples


def preprocess_frame(path, resolution):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if width < height:
        new_width = resolution
        new_height = int(round(height * resolution / width))
    else:
        new_height = resolution
        new_width = int(round(width * resolution / height))
    image = image.resize((new_width, new_height), RESAMPLE_BICUBIC)
    left = max((new_width - resolution) // 2, 0)
    top = max((new_height - resolution) // 2, 0)
    return image.crop((left, top, left + resolution, top + resolution))


def load_tactile_array(tactile_dir, frame_idx, resolution):
    path = Path(tactile_dir) / f"{frame_idx}.png"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(preprocess_frame(path, resolution), dtype=np.float32)


def contact_mask(
    residual,
    k=3.0,
    mask_dilate=3,
    min_threshold=1.0,
    threshold=None,
    percentile=None,
):
    score = np.mean(np.abs(residual), axis=-1)
    if threshold is None and percentile is not None:
        threshold = float(np.percentile(score, percentile))
    if threshold is None:
        median = float(np.median(score))
        mad = float(np.median(np.abs(score - median)))
        threshold = max(median + k * mad, float(min_threshold))
    threshold = max(float(threshold), float(min_threshold))
    return motion_contact_mask(
        residual,
        k=k,
        mask_dilate=mask_dilate,
        min_threshold=min_threshold,
        threshold=threshold,
    ), threshold


def pair2_dir(cache_dir, object_name, frame_a, frame_b):
    return Path(cache_dir) / object_name / f"pair2_{frame_a:05d}_{frame_b:05d}"


def save_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_png(path, array):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def motion_to_uint8(motion, motion_max, gamma):
    value = np.clip(motion / max(float(motion_max), 1e-6), 0.0, 1.0)
    value = value ** float(gamma)
    return np.clip(value * 255.0, 0, 255).astype(np.uint8)


def compute_pair(frame_a, frame_b, background, args, object_contact_threshold=None):
    residual_a = debaseline(frame_a, background)
    residual_b = debaseline(frame_b, background)
    percentile = args.contact_percentile if args.contact_threshold_mode == "frame_percentile" else None
    threshold = object_contact_threshold if args.contact_threshold_mode == "object_percentile" else None
    if args.contact_threshold_mode == "window_percentile":
        scores = [
            np.mean(np.abs(residual_a), axis=-1).reshape(-1),
            np.mean(np.abs(residual_b), axis=-1).reshape(-1),
        ]
        threshold = float(np.percentile(np.concatenate(scores), args.contact_percentile))
    mask_a, thr_a = contact_mask(
        residual_a, args.contact_k, args.mask_dilate, args.contact_min_threshold, threshold, percentile
    )
    mask_b, thr_b = contact_mask(
        residual_b, args.contact_k, args.mask_dilate, args.contact_min_threshold, threshold, percentile
    )
    mask = mask_a | mask_b
    raw_motion = np.mean(np.abs(residual_b - residual_a), axis=-1)
    if args.motion_median_filter:
        raw_motion = median_filter_3x3(raw_motion)
    motion_floor = float(np.percentile(raw_motion, args.motion_floor_percentile))
    motion_signal = np.clip(raw_motion - motion_floor, 0.0, None).astype(np.float32)

    inside = motion_signal[mask]
    outside = motion_signal[~mask]
    inside_mean = float(inside.mean()) if inside.size else 0.0
    outside_mean = float(outside.mean()) if outside.size else 0.0
    ratio = inside_mean / max(outside_mean, 1e-6)

    masked_motion = motion_signal * mask.astype(np.float32)
    pooled_motion = area_pool_2d(masked_motion, args.target_h, args.target_w)
    motion_map = np.clip(pooled_motion / max(args.motion_max, 1e-6), 0.0, 1.0) ** args.motion_gamma
    pooled_mask = area_pool_2d(mask.astype(np.float32), args.target_h, args.target_w)
    token_mask = pooled_mask >= args.mask_pool_threshold

    stats = {
        "threshold_a": float(thr_a),
        "threshold_b": float(thr_b),
        "mask_coverage_full": float(mask.mean()),
        "mask_coverage_27": float(token_mask.mean()),
        "motion_inside_mean": inside_mean,
        "motion_outside_mean": outside_mean,
        "in_out_motion_ratio": float(ratio),
        "motion_floor_percentile": float(args.motion_floor_percentile),
        "motion_floor": motion_floor,
        "raw_motion_mean": float(raw_motion.mean()),
        "raw_motion_p95": float(np.percentile(raw_motion, 95)),
        "motion_signal_mean": float(motion_signal.mean()),
        "dead_pair": bool(motion_signal.mean() <= args.dead_motion_mean_threshold),
    }
    return motion_map.astype(np.float32), token_mask.astype(np.uint8), masked_motion, mask, stats


def process_object(sample, args):
    object_cache_dir = Path(args.motion_cache_dir) / sample["object_name"]
    object_cache_dir.mkdir(parents=True, exist_ok=True)
    background = load_tactile_array(sample["tactile_dir"], 0, args.resolution)
    frames = {
        idx: load_tactile_array(sample["tactile_dir"], idx, args.resolution)
        for idx in sample["frame_indices"]
    }
    object_contact_threshold = None
    if args.contact_threshold_mode == "object_percentile":
        scores = [
            np.mean(np.abs(debaseline(frame, background)), axis=-1).reshape(-1)
            for frame in frames.values()
        ]
        object_contact_threshold = float(np.percentile(np.concatenate(scores), args.contact_percentile))
    pairs = list(zip(sample["frame_indices"][:-2], sample["frame_indices"][2:]))
    pair_stats = []

    for pair_index, (frame_a, frame_b) in enumerate(pairs, start=1):
        current_pair_dir = pair2_dir(args.motion_cache_dir, sample["object_name"], frame_a, frame_b)
        motion_path = current_pair_dir / "motion_map.npy"
        mask_path = current_pair_dir / "contact_mask.npy"
        meta_path = current_pair_dir / "pair_meta.json"
        if motion_path.exists() and mask_path.exists() and meta_path.exists() and not args.overwrite:
            stats = json.loads(meta_path.read_text())["stats"]
        else:
            current_pair_dir.mkdir(parents=True, exist_ok=True)
            motion_map, token_mask, _, _, stats = compute_pair(
                frames[frame_a], frames[frame_b], background, args, object_contact_threshold
            )
            np.save(motion_path, motion_map.astype(np.float32))
            np.save(mask_path, token_mask.astype(np.uint8))
            save_json(
                meta_path,
                {
                    "cache_version": CACHE_VERSION,
                    "object_name": sample["object_name"],
                    "frame_a": frame_a,
                    "frame_b": frame_b,
                    "pair_key": f"pair2_{frame_a:05d}_{frame_b:05d}",
                    "target_shape": [args.target_h, args.target_w],
                    "stats": stats,
                },
            )
        pair_stats.append({"frame_a": frame_a, "frame_b": frame_b, **stats})
        if pair_index % 50 == 0 or pair_index == len(pairs):
            print(f"  {sample['object_name']}: pair2 {pair_index}/{len(pairs)}")

    stats_by_pair = {
        f"{stat['frame_a']}_{stat['frame_b']}": stat
        for stat in pair_stats
    }
    dead_windows = 0
    for window in make_windows([sample], args.num_frames):
        _, _, selected = window
        first = stats_by_pair.get(f"{selected[0]}_{selected[2]}")
        second = stats_by_pair.get(f"{selected[2]}_{selected[4]}")
        if first and second and first["dead_pair"] and second["dead_pair"]:
            dead_windows += 1

    ratios = [stat["in_out_motion_ratio"] for stat in pair_stats]
    coverages = [stat["mask_coverage_27"] for stat in pair_stats]
    meta = {
        "cache_version": CACHE_VERSION,
        "object_name": sample["object_name"],
        "debaseline": "R_t = (I_t - B) - spatial_median(I_t - B)",
        "contact_threshold": {
            "formula": (
                "window_percentile: current diagnostic pair/window percentile over mean(abs(R_t), channel); "
                "object_percentile/frame_percentile: mean(abs(R_t), channel) > percentile; "
                "mad: mean(abs(R_t), channel) > max(median + k*MAD, min_threshold)"
            ),
            "mode": args.contact_threshold_mode,
            "k": args.contact_k,
            "percentile": args.contact_percentile,
            "object_threshold": object_contact_threshold,
            "min_threshold": args.contact_min_threshold,
            "mask_dilate": args.mask_dilate,
            "dilation": "PIL.ImageFilter.MaxFilter",
        },
        "motion": {
            "formula": "mean(abs(R_b - R_a), channel) * (C_a OR C_b)",
            "pair_stride": 2,
            "denoise": "3x3 median filter + pair-level motion floor subtraction",
            "motion_floor_percentile": args.motion_floor_percentile,
            "motion_max": args.motion_max,
            "motion_gamma": args.motion_gamma,
            "target_shape": [args.target_h, args.target_w],
            "mask_pool_threshold": args.mask_pool_threshold,
        },
        "split": sample["split"],
        "frame_split_mode": args.frame_split_mode,
        "ordered_indices": sample["frame_indices"],
        "pair_keys": [f"pair2_{a:05d}_{b:05d}" for a, b in pairs],
        "num_pairs": len(pair_stats),
        "dead_pair_count": int(sum(stat["dead_pair"] for stat in pair_stats)),
        "dead_window_count": dead_windows,
        "mask_coverage_27_mean": float(np.mean(coverages)) if coverages else 0.0,
        "mask_coverage_27_median": float(np.median(coverages)) if coverages else 0.0,
        "in_out_ratio_mean": float(np.mean(ratios)) if ratios else 0.0,
        "in_out_ratio_median": float(np.median(ratios)) if ratios else 0.0,
        "pair_stats": pair_stats,
    }
    save_json(object_cache_dir / "object_meta.json", meta)
    return meta


def make_windows(samples, num_frames):
    windows = []
    for sample in samples:
        indices = sample["frame_indices"]
        if len(indices) < num_frames:
            continue
        for start in range(len(indices) - num_frames + 1):
            windows.append((sample, start, indices[start:start + num_frames]))
    return windows


def to_rgb(gray):
    return np.repeat(gray[..., None], 3, axis=-1)


def draw_label(cell, text):
    image = Image.fromarray(cell)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 16), fill=(0, 0, 0))
    draw.text((3, 2), text, fill=(255, 255, 255))
    return np.asarray(image)


def save_compare_grid(path, rows):
    cell_h, cell_w = rows[0][1][0].shape[:2]
    label_w = 110
    grid = Image.new("RGB", (label_w + cell_w * 5, cell_h * len(rows)), (0, 0, 0))
    draw = ImageDraw.Draw(grid)
    for row_idx, (label, cells) in enumerate(rows):
        y = row_idx * cell_h
        draw.text((6, y + 6), label, fill=(255, 255, 255))
        for col_idx, cell in enumerate(cells):
            grid.paste(Image.fromarray(cell), (label_w + col_idx * cell_w, y))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def save_debug_window(sample, window_start, selected_indices, args):
    out_dir = Path(args.debug_output_dir) / sample["object_name"] / f"window_{window_start:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    object_contact_threshold = None
    meta_path = Path(args.motion_cache_dir) / sample["object_name"] / "object_meta.json"
    if args.contact_threshold_mode == "object_percentile" and meta_path.exists():
        object_contact_threshold = json.loads(meta_path.read_text())["contact_threshold"]["object_threshold"]
    background = load_tactile_array(sample["tactile_dir"], 0, args.resolution)
    frames = [load_tactile_array(sample["tactile_dir"], idx, args.resolution) for idx in selected_indices]
    images = [
        np.asarray(preprocess_frame(Path(sample["tactile_dir"]) / f"{idx}.png", args.resolution), dtype=np.uint8)
        for idx in selected_indices
    ]
    for idx, image in enumerate(images):
        save_png(out_dir / f"tactile_{idx:02d}.png", image)

    residuals = [debaseline(frame, background) for frame in frames]
    masks, _ = compute_contact_masks(
        residuals,
        contact_threshold_mode=args.contact_threshold_mode,
        contact_k=args.contact_k,
        contact_percentile=args.contact_percentile,
        mask_dilate=args.mask_dilate,
        contact_min_threshold=args.contact_min_threshold,
        object_threshold=object_contact_threshold,
    )
    motion_targets, motion_masks = compute_motion_targets(
        np.stack(frames, axis=0),
        background,
        target_h=args.target_h,
        target_w=args.target_w,
        contact_threshold_mode=args.contact_threshold_mode,
        contact_k=args.contact_k,
        contact_percentile=args.contact_percentile,
        object_threshold=object_contact_threshold,
        mask_dilate=args.mask_dilate,
        contact_min_threshold=args.contact_min_threshold,
        motion_max=args.motion_max,
        motion_gamma=args.motion_gamma,
        motion_floor_percentile=args.motion_floor_percentile,
        mask_pool_threshold=args.mask_pool_threshold,
        use_contact_mask=True,
        dead_motion_mean_threshold=args.dead_motion_mean_threshold,
    )
    for idx in (0, 2, 4):
        save_png(out_dir / f"contact_mask_{idx:02d}.png", masks[idx].astype(np.uint8) * 255)

    debug_motion = []
    debug_masked = []
    debug_stats = []
    for local_a, local_b in ((0, 2), (2, 4)):
        motion_map, token_mask, masked_motion, full_mask, stats = compute_pair(
            frames[local_a], frames[local_b], background, args, object_contact_threshold
        )
        motion_map = motion_targets[len(debug_motion)]
        token_mask = motion_masks[len(debug_motion)]
        frame_a = selected_indices[local_a]
        frame_b = selected_indices[local_b]
        motion_uint8 = motion_to_uint8(masked_motion, args.motion_max, args.motion_gamma)
        save_png(out_dir / f"motion_M{len(debug_motion)}_{frame_a:05d}_{frame_b:05d}.png", to_rgb(motion_uint8))
        np.save(out_dir / f"motion_M{len(debug_motion)}_{frame_a:05d}_{frame_b:05d}_27.npy", motion_map)
        np.save(out_dir / f"contact_mask_M{len(debug_motion)}_{frame_a:05d}_{frame_b:05d}_27.npy", token_mask)
        debug_motion.append(motion_uint8)
        debug_masked.append(full_mask)
        debug_stats.append({"frame_a": frame_a, "frame_b": frame_b, **stats})

    black = np.zeros_like(images[0])
    mask_row = [
        draw_label(to_rgb(masks[i].astype(np.uint8) * 255), f"C{i}") if i in (0, 2, 4) else black
        for i in range(5)
    ]
    motion_row = [
        draw_label(to_rgb(debug_motion[0]), "M0 0-2"),
        draw_label(to_rgb(debug_motion[0]), "M0"),
        draw_label(to_rgb(debug_motion[1]), "M1 2-4"),
        draw_label(to_rgb(debug_motion[1]), "M1"),
        draw_label(to_rgb(debug_motion[1]), "M1"),
    ]
    rows = [
        ("tactile", [draw_label(image, f"I{i}") for i, image in enumerate(images)]),
        ("contact", mask_row),
        ("motion", motion_row),
    ]
    save_compare_grid(out_dir / "compare_grid.png", rows)
    save_json(
        out_dir / "metadata.json",
        {
            "object_name": sample["object_name"],
            "window_start": window_start,
            "selected_indices": selected_indices,
            "pair_keys": [
                f"pair2_{selected_indices[0]:05d}_{selected_indices[2]:05d}",
                f"pair2_{selected_indices[2]:05d}_{selected_indices[4]:05d}",
            ],
            "stats": debug_stats,
            "dead_window": bool(all(stat["dead_pair"] for stat in debug_stats)),
            "visualization": {
                "motion_max": args.motion_max,
                "motion_gamma": args.motion_gamma,
                "mask_outside_forced_black": True,
                "note": "Black outside the mask is a display choice; target quality is judged by stats and metadata.",
            },
        },
    )


def collect_report(cache_dir):
    metas = []
    for path in sorted(Path(cache_dir).glob("*/object_meta.json")):
        metas.append(json.loads(path.read_text()))
    return metas


def summarize_report(metas):
    if not metas:
        return {
            "num_objects": 0,
            "num_pairs": 0,
            "mask_coverage_27_mean": 0.0,
            "in_out_ratio_median": 0.0,
            "dead_pair_count": 0,
            "dead_window_count": 0,
            "objects": [],
        }
    all_pairs = [stat for meta in metas for stat in meta.get("pair_stats", [])]
    ratios = [stat["in_out_motion_ratio"] for stat in all_pairs]
    covers = [stat["mask_coverage_27"] for stat in all_pairs]
    return {
        "num_objects": len(metas),
        "num_pairs": len(all_pairs),
        "mask_coverage_27_mean": float(np.mean(covers)) if covers else 0.0,
        "mask_coverage_27_median": float(np.median(covers)) if covers else 0.0,
        "in_out_ratio_mean": float(np.mean(ratios)) if ratios else 0.0,
        "in_out_ratio_median": float(np.median(ratios)) if ratios else 0.0,
        "dead_pair_count": int(sum(stat.get("dead_pair", False) for stat in all_pairs)),
        "dead_window_count": int(sum(meta.get("dead_window_count", 0) for meta in metas)),
        "objects": [
            {
                "object_name": meta["object_name"],
                "num_pairs": meta["num_pairs"],
                "mask_coverage_27_mean": meta["mask_coverage_27_mean"],
                "in_out_ratio_median": meta["in_out_ratio_median"],
                "dead_pair_count": meta["dead_pair_count"],
                "dead_window_count": meta["dead_window_count"],
            }
            for meta in metas
        ],
    }


def print_report(summary):
    print("\nMotion target report")
    print("--------------------")
    print(
        f"objects={summary['num_objects']} pairs={summary['num_pairs']} "
        f"mask27_mean={summary['mask_coverage_27_mean'] * 100:.2f}% "
        f"mask27_median={summary['mask_coverage_27_median'] * 100:.2f}% "
        f"in/out_median={summary['in_out_ratio_median']:.2f}x "
        f"dead_pairs={summary['dead_pair_count']} dead_windows={summary['dead_window_count']}"
    )
    for obj in summary["objects"]:
        print(
            f"  {obj['object_name']}: pairs={obj['num_pairs']} "
            f"mask27={obj['mask_coverage_27_mean'] * 100:.2f}% "
            f"in/out={obj['in_out_ratio_median']:.2f}x "
            f"dead_pairs={obj['dead_pair_count']} dead_windows={obj['dead_window_count']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--motion_cache_dir", type=Path, default=Path("outputs/motion_cache"))
    parser.add_argument("--resolution", type=int, default=432)
    parser.add_argument("--target_h", type=int, default=27)
    parser.add_argument("--target_w", type=int, default=27)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--split", choices=("train", "test", "all"), default="train")
    parser.add_argument("--frame_split_mode", choices=("contact_90_10", "legacy"), default="contact_90_10")
    parser.add_argument("--objects", nargs="*", default=None)
    parser.add_argument("--contact_threshold_mode", choices=("window_percentile", "mad", "frame_percentile", "object_percentile"), default="window_percentile")
    parser.add_argument("--contact_k", type=float, default=3.0)
    parser.add_argument("--contact_percentile", type=float, default=96.0)
    parser.add_argument("--contact_min_threshold", type=float, default=1.0)
    parser.add_argument("--mask_dilate", type=int, default=3)
    parser.add_argument("--mask_pool_threshold", type=float, default=0.25)
    parser.add_argument("--motion_max", type=float, default=32.0)
    parser.add_argument("--motion_gamma", type=float, default=0.5)
    parser.add_argument("--motion_median_filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--motion_floor_percentile", type=float, default=75.0)
    parser.add_argument("--dead_motion_mean_threshold", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_debug_vis", action="store_true")
    parser.add_argument("--debug_num_samples", type=int, default=20)
    parser.add_argument("--debug_output_dir", type=Path, default=Path("outputs/motion_debug"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--report_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.report_only:
        summary = summarize_report(collect_report(args.motion_cache_dir))
        print_report(summary)
        return

    args.motion_cache_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.data_root, args.csv_path, args.split, args.frame_split_mode, args.objects)
    if not samples:
        raise RuntimeError("No usable samples found. Check data_root/csv_path/split/objects.")

    print(f"Loaded {len(samples)} objects; writing motion cache to {args.motion_cache_dir}")
    object_metas = []
    for sample_index, sample in enumerate(samples, start=1):
        print(f"[{sample_index}/{len(samples)}] {sample['object_name']}: {len(sample['frame_indices'])} frames")
        object_metas.append(process_object(sample, args))

    summary = summarize_report(object_metas)
    run_meta = {
        "cache_version": CACHE_VERSION,
        "num_objects": len(object_metas),
        "objects": [meta["object_name"] for meta in object_metas],
        "zero_new_dependencies": True,
        "dependencies": ["numpy", "PIL"],
        "summary": summary,
    }
    save_json(args.motion_cache_dir / "run_meta.json", run_meta)
    save_json(args.motion_cache_dir / "run_report.json", summary)

    if args.save_debug_vis:
        windows = make_windows(samples, args.num_frames)
        rng = random.Random(args.seed)
        chosen = rng.sample(windows, k=min(args.debug_num_samples, len(windows)))
        print(f"Saving {len(chosen)} debug windows to {args.debug_output_dir}")
        for idx, (sample, window_start, selected_indices) in enumerate(chosen, start=1):
            print(f"  debug {idx}/{len(chosen)}: {sample['object_name']} {selected_indices}")
            save_debug_window(sample, window_start, selected_indices, args)

    print_report(summary)
    print("Done.")


if __name__ == "__main__":
    main()
