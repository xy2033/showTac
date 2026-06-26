#!/usr/bin/env python3
"""Small checks for tools/precompute_tactile_motion_target.py."""

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "precompute_tactile_motion_target.py"


def load_module():
    spec = importlib.util.spec_from_file_location("precompute_tactile_motion_target", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    m = load_module()

    background = np.full((54, 54, 3), 100, dtype=np.float32)
    frame = background.copy()
    frame[20:34, 20:34] += 30
    frame += 5

    residual = m.debaseline(frame, background)
    assert abs(float(np.median(residual[..., 0]))) < 1e-5

    contact, threshold = m.contact_mask(residual, k=3.0, mask_dilate=1)
    assert threshold > 0
    assert 0.04 < float(contact.mean()) < 0.12

    motion = np.zeros((54, 54), dtype=np.float32)
    motion[0:2, 0:2] = 1.0
    pooled = m.area_pool_2d(motion, 27, 27)
    assert pooled.shape == (27, 27)
    assert np.isclose(pooled[0, 0], 1.0)

    assert m.pair2_dir(Path("cache"), "obj", 3, 5) == Path("cache/obj/pair2_00003_00005")


if __name__ == "__main__":
    main()
