# coding=utf-8
# Copyright 2025 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tactile-Visual Dataset for Haptic Video Understanding & Generation.

Sequence structure:
    [BOS] {Physical Text Tokens} [BOV] {Visual Video Latents (clean)} [EOV]
    [BOV] {Target Tactile Video Latents (noised)} [EOV] [EOS]

The visual video serves as conditioning (always clean, t=1.0) and the tactile
video is the generation target (noised via flow matching, flow loss computed).
"""

import ast
import collections
import csv
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.utils import image_transform, format_sequence_tactile_gen
from datasets.tactile_motion import compute_motion_targets


class TactileVisualDataset(Dataset):
    """Dataset for tactile-video generation conditioned on visual video + text.

    Each sample consists of:
        - text_tokens: tokenized physical description (e.g. "Contact: True, Material: Rough Wood")
        - images: stacked tensor of shape (2, num_frames, C, H, W)
          images[0] = visual conditioning video frames (from img_gelsight/)
          images[1] = target tactile video frames (from gelsight/)
        - modality_positions: (2, 2) tensor with (offset, length) for each video segment
        - data_type: "tactile_visual_data"
    """

    def __init__(
            self,
            data_root: str,
            csv_path: str,
            text_tokenizer: Any,
            max_seq_len: int = 8192,
            image_size: int = 432,
            latent_height: int = 27,
            latent_width: int = 27,
            num_frames: int = 5,
            num_visual_tokens_per_frame: int = 729,
            num_tactile_tokens_per_frame: int = 729,
            num_visual_tokens: Optional[int] = None,
            num_tactile_tokens: Optional[int] = None,
            cond_dropout_prob: float = 0.1,
            split: str = "train",  # "train", "test", or "all"
            frame_split_mode: str = "contact_90_10",  # "legacy" or "contact_90_10"
            showo_token_ids: Optional[Dict[str, int]] = None,
            min_res: Optional[Tuple[int, int]] = None,
            clip_image_size: int = 384,
            motion_enabled: bool = False,
            use_contact_mask: bool = True,
            motion_contact_threshold_mode: str = "window_percentile",
            motion_contact_k: float = 3.0,
            motion_contact_percentile: float = 96.0,
            motion_max: float = 32.0,
            motion_gamma: float = 0.5,
    ) -> None:
        """
        Args:
            data_root: Root directory containing per-object subdirectories.
                       Each subdirectory has `img_gelsight/` (visual) and `gelsight/` (tactile).
            csv_path: Path to CSV with columns:
                      object_name, ..., text_description, full_sentence, train_indices, test_indices
            text_tokenizer: HuggingFace tokenizer for text processing.
            max_seq_len: Maximum total sequence length.
            image_size: Target resolution for frame resizing (square).
            latent_height: Height of VAE latent grid after patch embedding (H / 8 / patch_size).
            latent_width: Width of VAE latent grid after patch embedding (W / 8 / patch_size).
            num_frames: Number of frames to sample uniformly from each video.
            num_visual_tokens_per_frame: Tokens per frame in visual video segment.
            num_tactile_tokens_per_frame: Tokens per frame in tactile video segment.
            num_visual_tokens: Total tokens in the visual segment. When time embeddings
                               are enabled this includes the prepended time token.
            num_tactile_tokens: Total tokens in the tactile segment. When time embeddings
                                are enabled this includes the prepended time token.
            cond_dropout_prob: Probability of dropping text conditioning (CFG training).
            split: Which data split to use ("train", "test", or "all").
            frame_split_mode: Frame split strategy. "legacy" uses CSV train/test lists,
                              "contact_90_10" uses CSV columns 2/3 as a contiguous range
                              and splits it temporally into 90% train and 10% test.
            showo_token_ids: Dictionary mapping token names to their IDs.
            min_res: Minimum resolution filter (height, width).
            clip_image_size: Resolution for CLIP/SigLIP image transform.
        """
        self.data_root = data_root
        self.csv_path = csv_path
        self.text_tokenizer = text_tokenizer
        self.pad_id = self.text_tokenizer.pad_token_id
        self.bos_id = showo_token_ids['bos_id']
        self.eos_id = showo_token_ids['eos_id']
        self.bov_id = showo_token_ids['bov_id']
        self.eov_id = showo_token_ids['eov_id']
        self.vid_pad_id = showo_token_ids['vid_pad_id']
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.num_frames = num_frames
        self.num_visual_tokens_per_frame = num_visual_tokens_per_frame
        self.num_tactile_tokens_per_frame = num_tactile_tokens_per_frame
        self.num_visual_tokens = (
            num_visual_tokens
            if num_visual_tokens is not None
            else num_visual_tokens_per_frame * num_frames
        )
        self.num_tactile_tokens = (
            num_tactile_tokens
            if num_tactile_tokens is not None
            else num_tactile_tokens_per_frame * num_frames
        )
        self.h = latent_height
        self.w = latent_width
        self.cond_dropout_prob = cond_dropout_prob
        self.split = split
        self.frame_split_mode = frame_split_mode
        self.data_type = "tactile_visual_data"
        self.image_transform = image_transform
        self.clip_image_size = clip_image_size
        self.clip_mean = (0.5, 0.5, 0.5)
        self.clip_std = (0.5, 0.5, 0.5)
        self.motion_enabled = motion_enabled
        self.use_contact_mask = use_contact_mask
        self.motion_contact_threshold_mode = motion_contact_threshold_mode
        self.motion_contact_k = motion_contact_k
        self.motion_contact_percentile = motion_contact_percentile
        self.motion_max = motion_max
        self.motion_gamma = motion_gamma
        self._background_cache: Dict[str, np.ndarray] = {}

        if self.frame_split_mode not in {"legacy", "contact_90_10"}:
            raise ValueError(f"Unsupported frame_split_mode: {self.frame_split_mode}")
        if self.h * self.w != self.num_tactile_tokens_per_frame:
            raise ValueError(
                f"latent grid {self.h}x{self.w} != tactile tokens/frame {self.num_tactile_tokens_per_frame}"
            )

        # 4 for bos, eos, bov (x2), eov (x2)
        self.max_text_len = max_seq_len - self.num_visual_tokens - self.num_tactile_tokens - 6

        # Parse CSV to load samples
        self.samples: List[Dict[str, Any]] = []
        self._load_csv(split)

        print(f"TactileVisualDataset ({split} split) loaded. {len(self.samples)} samples!")
        print(f"  num_frames={num_frames}, visual_tokens={self.num_visual_tokens}, "
              f"tactile_tokens={self.num_tactile_tokens}, max_text_len={self.max_text_len}")

    @staticmethod
    def _parse_index_list(indices_str: str) -> List[int]:
        if not indices_str:
            return []
        try:
            parsed = ast.literal_eval(indices_str)
        except (SyntaxError, ValueError):
            return []
        if not isinstance(parsed, list):
            return []
        return [int(idx) for idx in parsed]

    @staticmethod
    def _resolve_legacy_split_indices(
            train_indices: List[int], test_indices: List[int], split: str
    ) -> List[int]:
        if split == "train":
            return train_indices
        if split == "test":
            return test_indices
        if split == "all":
            combined = train_indices + test_indices
            seen = set()
            deduped = []
            for idx in combined:
                if idx not in seen:
                    deduped.append(idx)
                    seen.add(idx)
            return deduped
        raise ValueError(f"Unsupported split: {split}")

    @staticmethod
    def _resolve_contact_split_indices(start_idx: int, end_idx: int, split: str) -> List[int]:
        if end_idx < start_idx:
            return []

        all_indices = list(range(start_idx, end_idx + 1))
        if split == "all":
            return all_indices

        total = len(all_indices)
        if total == 0:
            return []

        boundary = int(np.floor(total * 0.9))
        if total >= 2:
            boundary = min(max(boundary, 1), total - 1)
        else:
            boundary = total

        if split == "train":
            return all_indices[:boundary]
        if split == "test":
            return all_indices[boundary:]
        raise ValueError(f"Unsupported split: {split}")

    @staticmethod
    def _list_frame_numbers(frame_dir: str) -> List[int]:
        frame_numbers = []
        for filename in os.listdir(frame_dir):
            if not filename.endswith(".png"):
                continue
            stem = os.path.splitext(filename)[0]
            if stem.isdigit():
                frame_numbers.append(int(stem))
        return sorted(frame_numbers)

    def _filter_paired_frame_indices(
            self, indices: List[int], visual_dir: str, tactile_dir: str
    ) -> List[int]:
        visual_frames = set(self._list_frame_numbers(visual_dir))
        tactile_frames = set(self._list_frame_numbers(tactile_dir))
        paired_frames = visual_frames & tactile_frames
        return [idx for idx in indices if idx in paired_frames]

    def _load_csv(self, split: str) -> None:
        """Parse CSV file and resolve frame indices according to the configured split mode."""
        with open(self.csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 11:
                    continue

                object_name = row[0].strip()
                try:
                    contact_start_idx = int(row[1].strip())
                    contact_end_idx = int(row[2].strip())
                    text_description = row[-4].strip().strip("'\"")
                    full_sentence = row[-3].strip().strip("'\"")
                    train_indices_str = row[-2].strip()
                    test_indices_str = row[-1].strip()
                except (IndexError, ValueError):
                    continue

                train_indices = self._parse_index_list(train_indices_str)
                test_indices = self._parse_index_list(test_indices_str)

                # Check directories exist
                obj_dir = os.path.join(self.data_root, object_name)
                visual_dir = os.path.join(obj_dir, "img_gelsight")
                tactile_dir = os.path.join(obj_dir, "gelsight")
                if not os.path.isdir(visual_dir) or not os.path.isdir(tactile_dir):
                    print(f"Warning: skipping {object_name} - missing directories")
                    continue

                # Use the full_sentence as the text prompt
                text = full_sentence if full_sentence else text_description

                if self.frame_split_mode == "contact_90_10":
                    frame_indices = self._resolve_contact_split_indices(
                        contact_start_idx, contact_end_idx, split
                    )
                    frame_indices = self._filter_paired_frame_indices(
                        frame_indices, visual_dir, tactile_dir
                    )
                else:
                    frame_indices = self._resolve_legacy_split_indices(
                        train_indices, test_indices, split
                    )
                    frame_indices = self._filter_paired_frame_indices(
                        frame_indices, visual_dir, tactile_dir
                    )

                if split in {"train", "test"} and len(frame_indices) == 0:
                    print(
                        f"Warning: skipping {object_name} - no usable {split} frames "
                        f"after applying {self.frame_split_mode} split"
                    )
                    continue

                self.samples.append({
                    "object_name": object_name,
                    "visual_dir": visual_dir,
                    "tactile_dir": tactile_dir,
                    "text": text,
                    "text_description": text_description,
                    "contact_start_idx": contact_start_idx,
                    "contact_end_idx": contact_end_idx,
                    "train_indices": train_indices,
                    "test_indices": test_indices,
                    "frame_indices": frame_indices,
                    "split": split,
                })

    def _select_frame_indices(self, indices: List[int]) -> List[int]:
        """Select frame indices for one clip.

        For the temporal contact split, use contiguous clips to preserve motion continuity:
        random contiguous windows for train, deterministic leading windows for test/all.
        Legacy mode keeps the older uniform sampling behavior.
        """
        if len(indices) == 0:
            raise ValueError("No frame indices available for sampling")

        ordered_indices = sorted(indices)

        if len(ordered_indices) >= self.num_frames:
            if self.frame_split_mode == "contact_90_10":
                max_start = len(ordered_indices) - self.num_frames
                if self.split == "train" and max_start > 0:
                    start = random.randint(0, max_start)
                else:
                    start = 0
                end = start + self.num_frames
                return ordered_indices[start:end]

            selected_positions = np.linspace(
                0, len(ordered_indices) - 1, self.num_frames, dtype=int
            ).tolist()
            return [ordered_indices[pos] for pos in selected_positions]

        return ordered_indices + [ordered_indices[-1]] * (self.num_frames - len(ordered_indices))

    def _load_video_frames(
            self, frame_dir: str, indices: Optional[List[int]] = None, return_indices: bool = False
    ) -> Any:
        """Load and uniformly sample frames from a directory of PNG images.

        Args:
            frame_dir: Directory containing frame PNG files (named 0.png, 1.png, ...).
            indices: Specific frame indices to load. If None, uniformly sample num_frames.
            return_indices: Whether to also return the sampled frame numbers.

        Returns:
            List of PIL Images, and optionally the sampled frame numbers.
        """
        png_files = {
            int(os.path.splitext(filename)[0]): filename
            for filename in os.listdir(frame_dir)
            if filename.endswith(".png") and os.path.splitext(filename)[0].isdigit()
        }

        if len(png_files) == 0:
            raise ValueError(f"No PNG files found in {frame_dir}")

        available_indices = sorted(png_files)

        if indices is not None:
            filtered_indices = [idx for idx in indices if idx in png_files]
            if len(filtered_indices) == 0:
                raise ValueError(f"No requested frame indices exist in {frame_dir}")
            selected = self._select_frame_indices(filtered_indices)
        else:
            selected = self._select_frame_indices(available_indices)

        frames = []
        for idx in selected:
            frame_path = os.path.join(frame_dir, png_files[idx])
            frame = Image.open(frame_path).convert('RGB')
            frames.append(frame)

        if return_indices:
            return frames, selected
        return frames

    def load_sample_video(
            self,
            sample: Dict[str, Any],
            modality: str,
            selected_indices: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, List[int]]:
        if modality not in {"visual", "tactile"}:
            raise ValueError(f"Unsupported modality: {modality}")
        frame_dir = sample[f"{modality}_dir"]
        frames, selected_indices = self._load_video_frames(
            frame_dir,
            selected_indices if selected_indices is not None else sample["frame_indices"],
            return_indices=True
        )
        return self._process_frames(frames), selected_indices

    def _process_frames(self, frames: List[Image.Image]) -> torch.Tensor:
        """Transform a list of frames to a stacked tensor.

        Args:
            frames: List of PIL Images.

        Returns:
            Tensor of shape (num_frames, C, image_size, image_size).
        """
        processed = []
        for frame in frames:
            frame_tensor = self.image_transform(frame, resolution=self.image_size)
            processed.append(frame_tensor)
        return torch.stack(processed, dim=0)  # (T, C, H, W)

    def _frame_to_array(self, frame: "Image.Image") -> np.ndarray:
        """Match training/diagnostic resize + center crop for motion compute."""
        width, height = frame.size
        if width < height:
            new_width = self.image_size
            new_height = int(round(height * self.image_size / width))
        else:
            new_height = self.image_size
            new_width = int(round(width * self.image_size / height))
        if (new_width, new_height) != frame.size:
            resample = getattr(Image, "Resampling", Image).BICUBIC
            frame = frame.resize((new_width, new_height), resample)
        left = max((new_width - self.image_size) // 2, 0)
        top = max((new_height - self.image_size) // 2, 0)
        frame = frame.crop((left, top, left + self.image_size, top + self.image_size))
        return np.asarray(frame.convert("RGB"), dtype=np.float32)

    def _load_background(self, sample: Dict[str, Any]) -> np.ndarray:
        """Load and cache the no-contact reference frame (gelsight/0.png) per object."""
        obj = sample["object_name"]
        if obj not in self._background_cache:
            bg_path = os.path.join(sample["tactile_dir"], "0.png")
            if not os.path.exists(bg_path):
                raise FileNotFoundError(
                    f"motion_enabled requires no-contact reference {bg_path} (gelsight/0.png)."
                )
            self._background_cache[obj] = self._frame_to_array(Image.open(bg_path).convert("RGB"))
        return self._background_cache[obj]

    def _load_motion_targets(
        self, sample: Dict[str, Any], tactile_frames: List["Image.Image"]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (2,27,27) diff-motion targets + masks on-the-fly from the 5 frames."""
        if len(tactile_frames) != 5:
            raise ValueError(f"motion targets expect 5 frames, got {len(tactile_frames)}")
        frames_5 = np.stack([self._frame_to_array(f) for f in tactile_frames], axis=0)
        background = self._load_background(sample)
        motion, mask = compute_motion_targets(
            frames_5,
            background,
            target_h=self.h,
            target_w=self.w,
            contact_threshold_mode=self.motion_contact_threshold_mode,
            contact_k=self.motion_contact_k,
            contact_percentile=self.motion_contact_percentile,
            motion_max=self.motion_max,
            motion_gamma=self.motion_gamma,
            use_contact_mask=self.use_contact_mask,
        )
        return (
            torch.tensor(motion, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        try:
            sample = self.samples[idx]
            selected_indices = self._select_frame_indices(sample["frame_indices"])

            # Load visual video frames
            visual_tensor, _ = self.load_sample_video(sample, "visual", selected_indices)

            # Load tactile video frames
            tactile_frames, selected_indices = self._load_video_frames(
                sample["tactile_dir"], selected_indices, return_indices=True
            )
            tactile_tensor = self._process_frames(tactile_frames)

            # Stack as (2, T, C, H, W): [0]=visual, [1]=tactile
            images = torch.stack([visual_tensor, tactile_tensor], dim=0)  # (2, T, C, H, W)

            # Tokenize text with conditional dropout for CFG training
            text = sample["text"]
            if random.random() < self.cond_dropout_prob:
                text_tokens = self.text_tokenizer(
                    '', add_special_tokens=False, truncation=True,
                    max_length=self.max_text_len
                ).input_ids
            else:
                text_tokens = self.text_tokenizer(
                    text, add_special_tokens=False, truncation=True,
                    max_length=self.max_text_len
                ).input_ids

            # Build unified sequence with two [BOV]...[EOV] segments
            text_tokens, text_labels, modality_positions, text_mask, image_mask = \
                format_sequence_tactile_gen(
                    text_tokens=text_tokens,
                    bos_id=self.bos_id,
                    eos_id=self.eos_id,
                    bov_id=self.bov_id,
                    eov_id=self.eov_id,
                    pad_id=self.pad_id,
                    vid_pad_id=self.vid_pad_id,
                    num_visual_tokens=self.num_visual_tokens,
                    num_tactile_tokens=self.num_tactile_tokens,
                    max_seq_len=self.max_seq_len,
                )

            item = {
                'text_tokens': text_tokens,
                'text_labels': text_labels,
                'images': images,  # (2, T, C, H, W)
                'modality_positions': modality_positions,  # (2, 2)
                'text_masks': text_mask,
                'image_masks': image_mask,
                'texts': text,
                'object_names': sample["object_name"],
                'frame_indices': torch.tensor(selected_indices, dtype=torch.long),
                'data_type': self.data_type,
            }
            if self.motion_enabled:
                motion_targets, motion_mask = self._load_motion_targets(sample, tactile_frames)
                item['motion_targets'] = motion_targets
                item['motion_mask'] = motion_mask
            return item

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            if self.motion_enabled:
                raise
            return self.__getitem__((idx + 1) % len(self.samples))

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function that handles variable-length sequences."""
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('texts', 'data_type', 'object_names'):
                batched[k] = torch.stack(v, dim=0)
        return batched


if __name__ == '__main__':
    from models.misc import get_text_tokenizer

    text_tokenizer, showo_token_ids = get_text_tokenizer(
        "Qwen/Qwen2.5-1.5B-Instruct",
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name="qwen2_5"
    )

    dataset = TactileVisualDataset(
        data_root="/media/xy/Elements/tac/tacquad_gelsight_img/data_indoor",
        csv_path="/media/xy/Elements/tac/tacquad_gelsight_img/contact_indoor_list_tvl.csv",
        text_tokenizer=text_tokenizer,
        max_seq_len=8192,
        image_size=432,
        latent_height=27,
        latent_width=27,
        num_frames=5,
        num_visual_tokens_per_frame=729,
        num_tactile_tokens_per_frame=729,
        cond_dropout_prob=0.1,
        split="train",
        showo_token_ids=showo_token_ids,
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn, shuffle=True)

    for i, batch in enumerate(loader):
        print(f"Batch {i}:")
        print(f"  text_tokens: {batch['text_tokens'].shape}")
        print(f"  images: {batch['images'].shape}")
        print(f"  modality_positions: {batch['modality_positions'].shape}")
        print(f"  data_type: {batch['data_type']}")
        print(f"  modality_positions[0]: {batch['modality_positions'][0]}")
        if i >= 2:
            break
