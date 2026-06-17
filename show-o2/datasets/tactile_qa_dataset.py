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
Tactile QA Dataset for Stage 2 NTP (Next Token Prediction) supervision.

Extends TactileVisualDataset to build question→answer sequences where the
answer tokens receive valid text_labels for NTP loss, while the question and
video segments are masked (label=-100).

Sequence structure:
    [BOS] <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n
    [BOV] {Visual Video Latents (clean)} [EOV]
    [BOV] {Target Tactile Video Latents (noised)} [EOV]
    {answer}<|im_end|> [EOS]

Only answer tokens contribute to loss_ntp. Both visual and tactile video
tokens contribute to loss_flow as in pure generation.
"""

import csv
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch

from datasets.tactile_visual_dataset import TactileVisualDataset
from datasets.utils import format_sequence_tactile_qa


class TactileQADataset(TactileVisualDataset):
    """Dataset that pairs visual+tactile video with tactile attribute QA.

    Extends TactileVisualDataset, reusing video frame loading and preprocessing.
    Overrides CSV loading to read from a QA-formatted CSV and builds NTP-compatible
    sequences with question/answer token labelling.

    CSV format (tactile_qa_pairs.csv):
        object_name, question, answer, qa_type
    """

    def __init__(
            self,
            data_root: str,
            csv_path: str,                    # contact_indoor_list_tvl.csv (for frame info)
            tactile_qa_csv_path: str,         # tactile_qa_pairs.csv (for QA pairs)
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
            split: str = "train",
            frame_split_mode: str = "contact_90_10",
            showo_token_ids: Optional[Dict[str, int]] = None,
            min_res: Optional[Tuple[int, int]] = None,
            clip_image_size: int = 384,
    ) -> None:
        self.tactile_qa_csv_path = tactile_qa_csv_path
        # Call TactileVisualDataset.__init__ which sets up all infrastructure
        super().__init__(
            data_root=data_root,
            csv_path=csv_path,
            text_tokenizer=text_tokenizer,
            max_seq_len=max_seq_len,
            image_size=image_size,
            latent_height=latent_height,
            latent_width=latent_width,
            num_frames=num_frames,
            num_visual_tokens_per_frame=num_visual_tokens_per_frame,
            num_tactile_tokens_per_frame=num_tactile_tokens_per_frame,
            num_visual_tokens=num_visual_tokens,
            num_tactile_tokens=num_tactile_tokens,
            cond_dropout_prob=cond_dropout_prob,
            split=split,
            frame_split_mode=frame_split_mode,
            showo_token_ids=showo_token_ids,
            min_res=min_res,
            clip_image_size=clip_image_size,
        )
        self.data_type = "tactile_qa_data"

    def _load_csv(self, split: str) -> None:
        """Load QA pairs from tactile_qa_pairs.csv and match with video directories.

        Builds self.qa_pairs: list of dicts with object_name, question, answer, qa_type.
        Falls back to the parent's video frame index logic per object.
        """
        # First, build an object → contact indices mapping from the regular CSV
        obj_info: Dict[str, Dict[str, Any]] = {}
        with open(self.csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 11:
                    continue
                try:
                    obj_name = row[0].strip()
                    contact_start = int(row[1].strip())
                    contact_end = int(row[2].strip())
                    obj_dir = os.path.join(self.data_root, obj_name)
                    visual_dir = os.path.join(obj_dir, "img_gelsight")
                    tactile_dir = os.path.join(obj_dir, "gelsight")
                    if os.path.isdir(visual_dir) and os.path.isdir(tactile_dir):
                        obj_info[obj_name] = {
                            "object_name": obj_name,
                            "visual_dir": visual_dir,
                            "tactile_dir": tactile_dir,
                            "contact_start_idx": contact_start,
                            "contact_end_idx": contact_end,
                        }
                except (IndexError, ValueError):
                    continue

        # Now load QA pairs and match with object video directories
        self.samples: List[Dict[str, Any]] = []
        with open(self.tactile_qa_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                object_name = row.get("object_name", "").strip()
                question = row.get("question", "").strip()
                answer = row.get("answer", "").strip()
                qa_type = row.get("qa_type", "").strip()

                if not object_name or object_name not in obj_info:
                    continue

                info = obj_info[object_name]
                # Build frame indices from contact range
                all_indices = list(range(
                    info["contact_start_idx"], info["contact_end_idx"] + 1
                ))
                paired_indices = self._filter_paired_frame_indices(
                    all_indices, info["visual_dir"], info["tactile_dir"]
                )
                if not paired_indices:
                    continue

                # 90/10 train/test split (matches TactileVisualDataset behavior)
                total = len(paired_indices)
                if split == "all":
                    frame_indices = paired_indices
                elif split == "train":
                    boundary = min(max(int(total * 0.9), 1), total - 1) if total >= 2 else total
                    frame_indices = paired_indices[:boundary]
                else:
                    boundary = min(max(int(total * 0.9), 1), total - 1) if total >= 2 else total
                    frame_indices = paired_indices[boundary:]

                if not frame_indices:
                    continue

                self.samples.append({
                    "object_name": object_name,
                    "visual_dir": info["visual_dir"],
                    "tactile_dir": info["tactile_dir"],
                    "question": question,
                    "answer": answer,
                    "qa_type": qa_type,
                    "frame_indices": frame_indices,
                })

        print(f"TactileQADataset ({split} split) loaded. {len(self.samples)} QA samples!")

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        try:
            sample = self.samples[idx]
            selected_indices = self._select_frame_indices(sample["frame_indices"])

            # Load visual and tactile video frames
            visual_tensor, _ = self.load_sample_video(sample, "visual", selected_indices)
            tactile_frames, selected_indices = self._load_video_frames(
                sample["tactile_dir"], selected_indices, return_indices=True
            )
            tactile_tensor = self._process_frames(tactile_frames)
            images = torch.stack([visual_tensor, tactile_tensor], dim=0)  # (2, T, C, H, W)

            # Build chat-formatted question
            question_text = (
                f"<|im_start|>user\n{sample['question']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            answer_text = f"{sample['answer']}<|im_end|>"

            question_tokens = self.text_tokenizer(
                question_text, add_special_tokens=False,
                truncation=True, max_length=self.max_text_len // 2
            ).input_ids

            answer_tokens = self.text_tokenizer(
                answer_text, add_special_tokens=False,
                truncation=True, max_length=self.max_text_len // 2
            ).input_ids

            # Build QA sequence with proper token labelling
            text_tokens, text_labels, modality_positions, text_mask, image_mask = \
                format_sequence_tactile_qa(
                    question_tokens=question_tokens,
                    answer_tokens=answer_tokens,
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

            return {
                'text_tokens': text_tokens,
                'text_labels': text_labels,
                'images': images,
                'modality_positions': modality_positions,
                'text_masks': text_mask,
                'image_masks': image_mask,
                'texts': sample['answer'],
                'object_names': sample['object_name'],
                'data_type': self.data_type,
            }

        except Exception as e:
            print(f"Error loading QA sample {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))
