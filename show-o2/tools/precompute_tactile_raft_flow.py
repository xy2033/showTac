#!/usr/bin/env python3
"""Precompute adjacent-pair RAFT flow cache for tactile videos."""

import argparse
import ast
import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.transforms import functional as TVF


CACHE_VERSION = 1


@dataclass
class ObjectSample:
    object_name: str
    visual_dir: Path
    tactile_dir: Path
    frame_indices: List[int]
    contact_start_idx: int
    contact_end_idx: int
    split: str


def parse_index_list(value: str) -> List[int]:
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [int(idx) for idx in parsed]


def resolve_contact_split_indices(start_idx: int, end_idx: int, split: str) -> List[int]:
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


def resolve_legacy_split_indices(train_indices: List[int], test_indices: List[int], split: str) -> List[int]:
    if split == "train":
        return train_indices
    if split == "test":
        return test_indices
    if split != "all":
        raise ValueError(f"Unsupported split: {split}")
    seen = set()
    out = []
    for idx in train_indices + test_indices:
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out


def list_frame_numbers(frame_dir: Path) -> List[int]:
    return sorted(
        int(path.stem)
        for path in frame_dir.iterdir()
        if path.suffix == ".png" and path.stem.isdigit()
    )


def filter_paired_frame_indices(indices: Sequence[int], visual_dir: Path, tactile_dir: Path) -> List[int]:
    paired_frames = set(list_frame_numbers(visual_dir)) & set(list_frame_numbers(tactile_dir))
    return [idx for idx in indices if idx in paired_frames]


def load_samples(data_root: Path, csv_path: Path, split: str, frame_split_mode: str) -> List[ObjectSample]:
    samples = []
    with csv_path.open("r", newline="") as handle:
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

            object_dir = data_root / object_name
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

            if len(frame_indices) < 2:
                print(f"Warning: skipping {object_name} - fewer than 2 paired frames")
                continue

            samples.append(
                ObjectSample(
                    object_name=object_name,
                    visual_dir=visual_dir,
                    tactile_dir=tactile_dir,
                    frame_indices=sorted(frame_indices),
                    contact_start_idx=contact_start_idx,
                    contact_end_idx=contact_end_idx,
                    split=split,
                )
            )
    return samples


def preprocess_frame(path: Path, resolution: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    return transforms.CenterCrop((resolution, resolution))(image)


def load_tactile_frame(tactile_dir: Path, frame_idx: int, resolution: int) -> Image.Image:
    path = tactile_dir / f"{frame_idx}.png"
    if not path.exists():
        raise FileNotFoundError(path)
    return preprocess_frame(path, resolution)


def pair_dir(cache_dir: Path, object_name: str, frame_a: int, frame_b: int) -> Path:
    return cache_dir / object_name / f"pair_{frame_a:05d}_{frame_b:05d}"


def flow_to_rgb(flow: np.ndarray, max_mag: float = None) -> np.ndarray:
    """Convert HxWx2 flow to uint8 RGB with direction hue and magnitude value."""
    flow = np.nan_to_num(flow.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    dx = flow[..., 0]
    dy = flow[..., 1]
    mag = np.sqrt(dx * dx + dy * dy)
    if max_mag is None:
        max_mag = float(np.percentile(mag, 95))
    max_mag = max(float(max_mag), 1e-6)

    hue = (np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi)
    value = np.clip(mag / max_mag, 0.0, 1.0)
    saturation = value

    h6 = hue * 6.0
    sector = np.floor(h6).astype(np.int32) % 6
    frac = h6 - np.floor(h6)
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


def load_raft(device: torch.device):
    weights = Raft_Large_Weights.C_T_SKHT_V2
    model = raft_large(weights=weights, progress=True).to(device).eval()
    return model, weights.transforms(), str(weights)


@torch.inference_mode()
def compute_flow(
    model: torch.nn.Module,
    raft_transforms,
    frame_a: Image.Image,
    frame_b: Image.Image,
    device: torch.device,
) -> np.ndarray:
    image_a = TVF.to_tensor(frame_a).unsqueeze(0)
    image_b = TVF.to_tensor(frame_b).unsqueeze(0)
    image_a, image_b = raft_transforms(image_a, image_b)
    flow = model(image_a.to(device), image_b.to(device))[-1][0]
    return flow.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def pad_flow_sequence(flow_rgbs: List[np.ndarray], mode: str) -> List[np.ndarray]:
    if not flow_rgbs:
        raise ValueError("Cannot pad an empty flow sequence")
    if mode == "repeat_last":
        return flow_rgbs + [flow_rgbs[-1]]
    if mode == "zero_last":
        return flow_rgbs + [np.zeros_like(flow_rgbs[-1])]
    raise ValueError(f"Unsupported flow_padding_mode: {mode}")


def save_flow_grid(path: Path, tactile_frames: List[Image.Image], flow_frames: List[np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(tactile_frames), figsize=(2.2 * len(tactile_frames), 4.4))
    for idx, frame in enumerate(tactile_frames):
        axes[0, idx].imshow(frame)
        axes[0, idx].set_title(f"tactile {idx}")
        axes[0, idx].axis("off")
        axes[1, idx].imshow(flow_frames[idx])
        axes[1, idx].set_title(f"flow {idx}")
        axes[1, idx].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_flow_video(path: Path, flow_frames: List[np.ndarray], fps: int = 2) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(path, flow_frames, fps=fps, macro_block_size=1)
        return "imageio"
    except Exception as imageio_error:
        try:
            import cv2

            height, width = flow_frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            for frame in flow_frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            return "opencv-python"
        except Exception as cv2_error:
            raise RuntimeError(
                "Could not write mp4 with imageio or cv2. "
                f"imageio error: {imageio_error}; cv2 error: {cv2_error}"
            ) from cv2_error


def process_object(
    sample: ObjectSample,
    args: argparse.Namespace,
    model: torch.nn.Module,
    raft_transforms,
    device: torch.device,
    weights_name: str,
) -> dict:
    object_cache_dir = Path(args.raft_cache_dir) / sample.object_name
    object_cache_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(zip(sample.frame_indices[:-1], sample.frame_indices[1:]))
    magnitudes = []
    for pair_index, (frame_a, frame_b) in enumerate(pairs, start=1):
        current_pair_dir = pair_dir(Path(args.raft_cache_dir), sample.object_name, frame_a, frame_b)
        raw_path = current_pair_dir / "raw_flow.npy"
        if raw_path.exists() and not args.overwrite:
            flow = np.load(raw_path)
        else:
            current_pair_dir.mkdir(parents=True, exist_ok=True)
            image_a = load_tactile_frame(sample.tactile_dir, frame_a, args.resolution)
            image_b = load_tactile_frame(sample.tactile_dir, frame_b, args.resolution)
            flow = compute_flow(model, raft_transforms, image_a, image_b, device)
            np.save(raw_path, flow)
        magnitudes.append(np.sqrt(np.sum(flow * flow, axis=-1)).reshape(-1))
        if pair_index % 25 == 0 or pair_index == len(pairs):
            print(f"  {sample.object_name}: raw flow {pair_index}/{len(pairs)}")

    max_mag = float(np.percentile(np.concatenate(magnitudes), 95))
    max_mag = max(max_mag, 1e-6)

    for frame_a, frame_b in pairs:
        current_pair_dir = pair_dir(Path(args.raft_cache_dir), sample.object_name, frame_a, frame_b)
        flow_rgb_path = current_pair_dir / "flow_rgb.npy"
        if flow_rgb_path.exists() and not args.overwrite:
            continue
        flow = np.load(current_pair_dir / "raw_flow.npy")
        np.save(flow_rgb_path, flow_to_rgb(flow, max_mag=max_mag))

    meta = {
        "cache_version": CACHE_VERSION,
        "object_name": sample.object_name,
        "normalization": "object_level_p95",
        "max_mag": max_mag,
        "resolution": args.resolution,
        "flow_resolution": [args.resolution, args.resolution],
        "flow_resolution_policy": "training_aligned_resize_centercrop_before_raft",
        "flow_padding_mode": args.flow_padding_mode,
        "frame_split_mode": args.frame_split_mode,
        "split": sample.split,
        "ordered_indices": sample.frame_indices,
        "num_pairs": len(pairs),
        "pair_keys": [f"pair_{a:05d}_{b:05d}" for a, b in pairs],
        "raft_weights": weights_name,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "vae_time_compression_note": "WanVAE compresses 5 flow RGB frames to 2 latent frames; this target is a coarse motion regularizer.",
    }
    save_json(object_cache_dir / "object_meta.json", meta)
    return meta


def make_debug_windows(samples: List[ObjectSample], num_frames: int) -> List[Tuple[ObjectSample, int, List[int]]]:
    windows = []
    for sample in samples:
        indices = sample.frame_indices
        if len(indices) < num_frames:
            continue
        for start in range(len(indices) - num_frames + 1):
            windows.append((sample, start, indices[start:start + num_frames]))
    return windows


def save_debug_window(
    sample: ObjectSample,
    window_start: int,
    selected_indices: List[int],
    args: argparse.Namespace,
) -> None:
    object_cache_dir = Path(args.raft_cache_dir) / sample.object_name
    with (object_cache_dir / "object_meta.json").open("r") as handle:
        object_meta = json.load(handle)

    out_dir = Path(args.debug_output_dir) / sample.object_name / f"window_{window_start:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tactile_frames = [
        load_tactile_frame(sample.tactile_dir, frame_idx, args.resolution)
        for frame_idx in selected_indices
    ]
    for idx, frame in enumerate(tactile_frames):
        frame.save(out_dir / f"tactile_{idx:02d}.png")

    flow_rgbs = []
    for idx, (frame_a, frame_b) in enumerate(zip(selected_indices[:-1], selected_indices[1:])):
        flow_path = pair_dir(Path(args.raft_cache_dir), sample.object_name, frame_a, frame_b) / "flow_rgb.npy"
        if not flow_path.exists():
            raise FileNotFoundError(f"Missing cached flow: {flow_path}")
        flow_rgb = np.load(flow_path)
        flow_rgbs.append(flow_rgb)
        save_png(out_dir / f"flow_{idx:02d}_to_{idx + 1:02d}_rgb.png", flow_rgb)

    padded_flows = pad_flow_sequence(flow_rgbs, args.flow_padding_mode)
    pad_name = "repeat_last" if args.flow_padding_mode == "repeat_last" else "zero_last"
    save_png(out_dir / f"flow_04_pad_{pad_name}_rgb.png", padded_flows[-1])
    save_flow_grid(out_dir / "flow_grid.png", tactile_frames, padded_flows)
    video_backend = save_flow_video(out_dir / "flow_video.mp4", padded_flows)

    metadata = {
        "object_name": sample.object_name,
        "window_start": window_start,
        "selected_indices": selected_indices,
        "normalization": object_meta["normalization"],
        "max_mag": object_meta["max_mag"],
        "padding_mode": args.flow_padding_mode,
        "num_frames": args.num_frames,
        "num_raw_flows": args.num_frames - 1,
        "flow_frame_mapping": {
            "flow_rgb_0": "frame_0_to_frame_1",
            "flow_rgb_1": "frame_1_to_frame_2",
            "flow_rgb_2": "frame_2_to_frame_3",
            "flow_rgb_3": "frame_3_to_frame_4",
            "flow_rgb_4": "repeat_of_frame_3_to_frame_4"
            if args.flow_padding_mode == "repeat_last"
            else "zero_padding",
        },
        "video_backend": video_backend,
        "tactile_frame_policy": "training_aligned_resize_centercrop",
        "vae_time_compression_note": object_meta["vae_time_compression_note"],
    }
    save_json(out_dir / "metadata.json", metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--csv_path", required=True, type=Path)
    parser.add_argument("--raft_cache_dir", required=True, type=Path)
    parser.add_argument("--resolution", type=int, default=432)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--split", choices=("train", "test", "all"), default="train")
    parser.add_argument("--frame_split_mode", choices=("contact_90_10", "legacy"), default="contact_90_10")
    parser.add_argument("--flow_padding_mode", choices=("repeat_last", "zero_last"), default="repeat_last")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_debug_vis", action="store_true")
    parser.add_argument("--debug_num_samples", type=int, default=20)
    parser.add_argument("--debug_output_dir", type=Path, default=Path("outputs/flow_debug"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.raft_cache_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.data_root, args.csv_path, args.split, args.frame_split_mode)
    if not samples:
        raise RuntimeError("No usable samples found. Check data_root/csv_path/split.")

    device = torch.device(args.device)
    model, raft_transforms, weights_name = load_raft(device)
    print(f"Loaded {len(samples)} objects; writing cache to {args.raft_cache_dir}")
    print(f"RAFT weights: {weights_name}; device: {device}")

    object_metas = []
    for sample_index, sample in enumerate(samples, start=1):
        print(f"[{sample_index}/{len(samples)}] {sample.object_name}: {len(sample.frame_indices)} frames")
        object_metas.append(process_object(sample, args, model, raft_transforms, device, weights_name))

    run_meta = {
        "cache_version": CACHE_VERSION,
        "num_objects": len(samples),
        "objects": [meta["object_name"] for meta in object_metas],
        "normalization": "object_level_p95",
        "resolution": args.resolution,
        "flow_padding_mode": args.flow_padding_mode,
        "split": args.split,
        "frame_split_mode": args.frame_split_mode,
        "raft_weights": weights_name,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }
    save_json(args.raft_cache_dir / "run_meta.json", run_meta)

    if args.save_debug_vis:
        windows = make_debug_windows(samples, args.num_frames)
        rng = random.Random(args.seed)
        chosen = rng.sample(windows, k=min(args.debug_num_samples, len(windows)))
        print(f"Saving {len(chosen)} debug windows to {args.debug_output_dir}")
        for idx, (sample, window_start, selected_indices) in enumerate(chosen, start=1):
            print(f"  debug {idx}/{len(chosen)}: {sample.object_name} {selected_indices}")
            save_debug_window(sample, window_start, selected_indices, args)

    print("Done.")


if __name__ == "__main__":
    main()
