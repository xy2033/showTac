#!/usr/bin/env python3
"""Regression checks for Task 2-5 motion auxiliary (v3 on-the-fly)."""

import tempfile
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.tactile_motion import compute_motion_targets
from datasets.tactile_visual_dataset import TactileVisualDataset


class DummyTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, truncation=True, max_length=None):
        class Result:
            input_ids = [11, 12, 13]

        return Result()


TOKEN_IDS = {
    "bos_id": 1,
    "eos_id": 2,
    "bov_id": 3,
    "eov_id": 4,
    "vid_pad_id": 5,
}


def write_png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def _make_object(data_root, object_name, with_contact=True):
    """Write 0.png (background) + 5 tactile frames with a moving contact blob."""
    res = 64
    bg = np.full((res, res, 3), 100, dtype=np.uint8)
    write_png(data_root / object_name / "gelsight" / "0.png", bg)
    write_png(data_root / object_name / "img_gelsight" / "0.png", bg)
    for idx in range(5):
        frame = bg.astype(np.float32)
        if with_contact:
            cy, cx = 20 + idx * 3, 25 + idx * 3
            frame[cy - 8:cy + 8, cx - 8:cx + 8] += 60.0
        write_png(data_root / object_name / "gelsight" / f"{idx}.png", np.clip(frame, 0, 255))
        write_png(data_root / object_name / "img_gelsight" / f"{idx}.png", np.clip(frame, 0, 255))


def make_dataset(tmp, motion_enabled=True, use_contact_mask=True, with_contact=True):
    data_root = tmp / "data"
    _make_object(data_root, "toy_object", with_contact=with_contact)
    csv_path = tmp / "contact.csv"
    csv_path.write_text(
        "toy_object,0,4,x,x,x,x,'desc','full','[0,1,2,3,4]','[0,1,2,3,4]'\n"
    )
    return TactileVisualDataset(
        data_root=str(data_root),
        csv_path=str(csv_path),
        text_tokenizer=DummyTokenizer(),
        max_seq_len=1600,
        image_size=54,            # divisible by 27 -> area_pool works
        latent_height=27,
        latent_width=27,
        num_frames=5,
        num_visual_tokens_per_frame=729,
        num_tactile_tokens_per_frame=729,
        num_visual_tokens=729,
        num_tactile_tokens=729,
        cond_dropout_prob=0.0,
        split="all",
        showo_token_ids=TOKEN_IDS,
        motion_enabled=motion_enabled,
        use_contact_mask=use_contact_mask,
    )


def test_dataset_unchanged_when_motion_disabled():
    with tempfile.TemporaryDirectory() as td:
        item = make_dataset(Path(td), motion_enabled=False)[0]
        assert "motion_targets" not in item
        assert "motion_mask" not in item


def test_dataset_computes_motion_on_the_fly():
    with tempfile.TemporaryDirectory() as td:
        item = make_dataset(Path(td))[0]
        assert item["motion_targets"].shape == (2, 27, 27)
        assert item["motion_mask"].shape == (2, 27, 27)
        # motion present and bounded; mask is a subset (contact-aware)
        assert float(item["motion_targets"].max()) > 0.0
        assert float(item["motion_targets"].min()) >= 0.0
        assert float(item["motion_targets"].max()) <= 1.0
        cov = float(item["motion_mask"].mean())
        assert 0.0 < cov < 1.0, f"unexpected mask coverage {cov}"
        # token-aligned: no motion outside accepted contact tokens
        m, msk = item["motion_targets"].numpy(), item["motion_mask"].numpy()
        assert m[msk == 0].max() == 0.0


def test_window_percentile_default_rejects_low_mad_background():
    h = w = 54
    background = np.full((h, w, 3), 100, dtype=np.float32)
    frames = np.stack([background.copy() for _ in range(5)], axis=0)
    # Broad weak residual: MAD mode with min_threshold=1 accepts this whole area.
    frames[:, :, :24] += 2.0
    # Small moving contact: window p96 should keep this stronger region.
    for idx in range(5):
        y0 = 15 + idx
        x0 = 28 + idx
        frames[idx, y0:y0 + 10, x0:x0 + 10] += 30.0

    default_motion, default_mask = compute_motion_targets(frames, background)
    mad_motion, mad_mask = compute_motion_targets(
        frames,
        background,
        contact_threshold_mode="mad",
        contact_k=3.0,
    )

    default_cov = float(default_mask.mean())
    mad_cov = float(mad_mask.mean())
    assert 0.0 < default_cov < 0.20
    assert mad_cov > default_cov * 2.0
    assert float(default_motion.max()) > 0.0


def test_ablation_disables_contact_mask():
    with tempfile.TemporaryDirectory() as td:
        item = make_dataset(Path(td), use_contact_mask=False)[0]
        assert item["motion_mask"].shape == (2, 27, 27)
        assert item["motion_mask"].min().item() == 1
        assert item["motion_mask"].max().item() == 1


def test_dead_window_zeroed():
    with tempfile.TemporaryDirectory() as td:
        item = make_dataset(Path(td), with_contact=False)[0]
        # no contact -> mask empty, target zero (loss contributes 0)
        assert float(item["motion_mask"].sum()) == 0.0
        assert float(item["motion_targets"].sum()) == 0.0


def test_missing_background_raises():
    with tempfile.TemporaryDirectory() as td:
        dataset = make_dataset(Path(td))
        bg = Path(td) / "data" / "toy_object" / "gelsight" / "0.png"
        bg.unlink()
        try:
            dataset[0]
        except FileNotFoundError as exc:
            assert "0.png" in str(exc)
        else:
            raise AssertionError("expected FileNotFoundError for missing background")


def test_no_cache_attributes():
    with tempfile.TemporaryDirectory() as td:
        dataset = make_dataset(Path(td))
        assert not hasattr(dataset, "motion_cache_dir")
        assert not hasattr(dataset, "motion_stride")


def test_force_paths_replaced_in_core_files():
    files = [
        ROOT / "datasets" / "tactile_visual_dataset.py",
        ROOT / "models" / "modeling_showo2_qwen2_5.py",
        ROOT / "train_tactile_stage_one.py",
        ROOT / "configs" / "showo2_1.5b_tactile_stage_one.yaml",
        ROOT / "run_tactile_stage_one.py",
    ]
    forbidden = ["virtual_force", "tactile_force_head", "loss_force", "force_pred", "force_proxy"]
    for path in files:
        text = path.read_text()
        for token in forbidden:
            assert token not in text, f"{token} still present in {path}"

    model_text = (ROOT / "models" / "modeling_showo2_qwen2_5.py").read_text()
    assert "tactile_motion_head" in model_text
    assert "visual_motion_predictor" in model_text
    assert "motion_token_encoder" in model_text
    assert "motion_injection_logit" in model_text
    assert "self.tactile_motion_head(tactile_hidden)" not in model_text
    assert "motion_condition" in model_text
    assert "aux_only" in model_text
    assert "baseline" in model_text

    runner_text = (ROOT / "run_tactile_stage_one.py").read_text()
    assert "MOTION_MODE" in runner_text
    assert "motion_aux.mode" in runner_text
    assert "MAX_TRAIN_STEPS" in runner_text
    assert "TRAIN_OUTPUT_DIR" in runner_text
    assert "motion_aux.enabled" in runner_text
    assert "motion_aux.coeff" in runner_text
    assert "motion_aux.contact_threshold_mode" in runner_text
    assert "motion_aux.contact_percentile" in runner_text

    train_text = (ROOT / "train_tactile_stage_one.py").read_text()
    assert "check_motion_condition_grads" in train_text
    assert "motion_grad/visual_motion_predictor" in train_text
    assert "motion_grad/motion_token_encoder" in train_text
    assert "motion_grad/motion_injection_gate" in train_text

    inference_text = (ROOT / "run_tactile_inference.py").read_text()
    assert "MOTION_MODE" in inference_text
    assert "--motion_mode" in inference_text

    config_text = (ROOT / "configs" / "showo2_1.5b_tactile_stage_one.yaml").read_text()
    assert "mode: motion_condition" in config_text
    assert "contact_threshold_mode: window_percentile" in config_text
    assert "contact_percentile: 96.0" in config_text

    # no cache references remain
    for path in [ROOT / "datasets" / "tactile_visual_dataset.py",
                 ROOT / "train_tactile_stage_one.py",
                 ROOT / "configs" / "showo2_1.5b_tactile_stage_one.yaml"]:
        assert "motion_cache_dir" not in path.read_text(), f"motion_cache_dir still in {path}"


if __name__ == "__main__":
    test_dataset_unchanged_when_motion_disabled()
    test_dataset_computes_motion_on_the_fly()
    test_window_percentile_default_rejects_low_mad_background()
    test_ablation_disables_contact_mask()
    test_dead_window_zeroed()
    test_missing_background_raises()
    test_no_cache_attributes()
    test_force_paths_replaced_in_core_files()
    print("ALL TASK2-5 v3 TESTS PASSED")
