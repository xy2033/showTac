"""Contact-aware diff-motion target computation (shared by dataset & diagnostics).

v3: targets are computed on-the-fly during training from the 5 sampled tactile
frames + the no-contact background frame. No precomputed cache, no disk I/O.

Pipeline (per 5-frame window I0..I4, background B):
  R_t = (I_t - B) - spatial_median(I_t - B)          # debaseline global drift
  C_t = |R_t|.mean(ch) > window_adaptive_thr          # contact mask (+ dilation)
  M0 = |R2 - R0|.mean(ch),  M1 = |R4 - R2|.mean(ch)   # two long-baseline motions
  M  = clip(median3x3(M) - p75_floor, 0)              # denoise + floor
  M0 *= (C0|C2),  M1 *= (C2|C4)                        # dual gating
  motion = area_pool(clip(M/motion_max,0,1)^gamma)    # normalize + downsample 27x27
  mask   = area_pool(C) >= mask_pool_threshold

All ops are numpy + PIL only (zero new deps).
"""

import numpy as np
from PIL import Image, ImageFilter


def debaseline(frame, background):
    """Subtract background and remove per-frame global low-frequency drift."""
    residual = frame.astype(np.float32) - background.astype(np.float32)
    lowfreq = np.median(residual, axis=(0, 1), keepdims=True)
    return residual - lowfreq


def _dilate_mask(mask, size):
    if size <= 1:
        return mask.astype(bool)
    if size % 2 == 0:
        raise ValueError(f"mask_dilate must be odd, got {size}")
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return np.asarray(image.filter(ImageFilter.MaxFilter(size)), dtype=np.uint8) > 0


def contact_mask(
    residual,
    k=3.0,
    mask_dilate=3,
    min_threshold=1.0,
    threshold=None,
    percentile=None,
):
    """Contact mask from a residual frame.

    If threshold is provided, use it directly. If percentile is provided, use
    that frame percentile. Otherwise fall back to median + k*MAD.
    """
    score = np.mean(np.abs(residual), axis=-1)
    if threshold is None and percentile is not None:
        threshold = float(np.percentile(score, percentile))
    if threshold is None:
        median = float(np.median(score))
        mad = float(np.median(np.abs(score - median)))
        threshold = median + k * mad
    threshold = max(float(threshold), float(min_threshold))
    return _dilate_mask(score > threshold, mask_dilate)


def compute_contact_masks(
    residuals,
    *,
    contact_threshold_mode="window_percentile",
    contact_k=3.0,
    contact_percentile=96.0,
    object_threshold=None,
    mask_dilate=3,
    contact_min_threshold=1.0,
):
    """Return one contact mask per residual using a shared threshold policy."""
    mode = contact_threshold_mode
    if mode == "window_percentile":
        scores = [np.mean(np.abs(r), axis=-1).reshape(-1) for r in residuals]
        threshold = float(np.percentile(np.concatenate(scores), contact_percentile))
        threshold = max(threshold, float(contact_min_threshold))
        masks = [
            contact_mask(r, contact_k, mask_dilate, contact_min_threshold, threshold=threshold)
            for r in residuals
        ]
        return masks, [threshold] * len(residuals)
    if mode == "frame_percentile":
        masks, thresholds = [], []
        for r in residuals:
            score = np.mean(np.abs(r), axis=-1)
            threshold = max(float(np.percentile(score, contact_percentile)), float(contact_min_threshold))
            masks.append(contact_mask(r, contact_k, mask_dilate, contact_min_threshold, threshold=threshold))
            thresholds.append(threshold)
        return masks, thresholds
    if mode == "object_percentile":
        if object_threshold is None:
            raise ValueError("object_threshold is required for object_percentile contact masks")
        threshold = max(float(object_threshold), float(contact_min_threshold))
        masks = [
            contact_mask(r, contact_k, mask_dilate, contact_min_threshold, threshold=threshold)
            for r in residuals
        ]
        return masks, [threshold] * len(residuals)
    if mode == "mad":
        masks, thresholds = [], []
        for r in residuals:
            score = np.mean(np.abs(r), axis=-1)
            median = float(np.median(score))
            mad = float(np.median(np.abs(score - median)))
            threshold = max(median + contact_k * mad, float(contact_min_threshold))
            masks.append(contact_mask(r, contact_k, mask_dilate, contact_min_threshold, threshold=threshold))
            thresholds.append(threshold)
        return masks, thresholds
    raise ValueError(f"Unsupported contact_threshold_mode: {contact_threshold_mode}")


def median_filter_3x3(value):
    padded = np.pad(value, 1, mode="edge")
    windows = [
        padded[y:y + value.shape[0], x:x + value.shape[1]]
        for y in range(3)
        for x in range(3)
    ]
    return np.median(np.stack(windows, axis=0), axis=0).astype(np.float32)


def area_pool_2d(value, out_h, out_w):
    h, w = value.shape
    if h % out_h != 0 or w % out_w != 0:
        raise ValueError(f"Input {value.shape} not divisible by {(out_h, out_w)}")
    return value.reshape(out_h, h // out_h, out_w, w // out_w).mean(axis=(1, 3)).astype(np.float32)


def _one_pair(residual_a, residual_b, mask, *, motion_max, motion_gamma,
              motion_floor_percentile, target_h, target_w, mask_pool_threshold):
    raw_motion = median_filter_3x3(np.mean(np.abs(residual_b - residual_a), axis=-1))
    floor = float(np.percentile(raw_motion, motion_floor_percentile))
    motion_signal = np.clip(raw_motion - floor, 0.0, None).astype(np.float32)
    masked = motion_signal * mask.astype(np.float32)
    pooled = area_pool_2d(masked, target_h, target_w)
    motion_map = np.clip(pooled / max(motion_max, 1e-6), 0.0, 1.0) ** motion_gamma
    token_mask = area_pool_2d(mask.astype(np.float32), target_h, target_w) >= mask_pool_threshold
    # Keep target and mask token-aligned: zero motion outside accepted contact tokens.
    motion_map = motion_map * token_mask.astype(np.float32)
    return motion_map.astype(np.float32), token_mask.astype(np.uint8), float(motion_signal.mean())


def compute_motion_targets(
    frames_5,
    background,
    *,
    target_h=27,
    target_w=27,
    contact_threshold_mode="window_percentile",
    contact_k=3.0,
    contact_percentile=96.0,
    object_threshold=None,
    mask_dilate=3,
    contact_min_threshold=1.0,
    motion_max=32.0,
    motion_gamma=0.5,
    motion_floor_percentile=75.0,
    mask_pool_threshold=0.25,
    use_contact_mask=True,
    dead_motion_mean_threshold=1e-4,
):
    """Compute (2,27,27) motion targets + masks from a 5-frame window.

    Args:
        frames_5:   (5, H, W, 3) float32, sampled tactile frames (window order).
        background: (H, W, 3) float32, no-contact reference (gelsight/0.png).
    Returns:
        motion_targets: (2, target_h, target_w) float32 in [0, 1]
        motion_mask:    (2, target_h, target_w) uint8 {0, 1}  (all-ones if use_contact_mask=False)
    """
    frames_5 = np.asarray(frames_5, dtype=np.float32)
    if frames_5.shape[0] != 5:
        raise ValueError(f"compute_motion_targets expects 5 frames, got {frames_5.shape[0]}")
    residuals = [debaseline(frames_5[t], background) for t in range(5)]
    full_gate = np.ones(frames_5.shape[1:3], dtype=bool)
    if use_contact_mask:
        masks, _ = compute_contact_masks(
            residuals,
            contact_threshold_mode=contact_threshold_mode,
            contact_k=contact_k,
            contact_percentile=contact_percentile,
            mask_dilate=mask_dilate,
            contact_min_threshold=contact_min_threshold,
            object_threshold=object_threshold,
        )
        gates = (masks[0] | masks[2], masks[2] | masks[4])
    else:
        # Ablation: no contact gating -> full-frame motion map + all-ones mask.
        gates = (full_gate, full_gate)

    # Two long-baseline pairs aligned to the 2 VAE latent frames.
    pair_specs = ((0, 2, gates[0]), (2, 4, gates[1]))
    motion_list, mask_list = [], []
    for a, b, gate in pair_specs:
        motion_map, token_mask, sig_mean = _one_pair(
            residuals[a], residuals[b], gate,
            motion_max=motion_max, motion_gamma=motion_gamma,
            motion_floor_percentile=motion_floor_percentile,
            target_h=target_h, target_w=target_w, mask_pool_threshold=mask_pool_threshold,
        )
        # Dead window: negligible motion -> zero target + empty mask (loss contributes 0).
        if use_contact_mask and sig_mean <= dead_motion_mean_threshold:
            motion_map = np.zeros((target_h, target_w), dtype=np.float32)
            token_mask = np.zeros((target_h, target_w), dtype=np.uint8)
        motion_list.append(motion_map)
        mask_list.append(token_mask if use_contact_mask else np.ones((target_h, target_w), dtype=np.uint8))

    return np.stack(motion_list, axis=0), np.stack(mask_list, axis=0)
