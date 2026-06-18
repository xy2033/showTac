#!/bin/bash
set -e

# =============================================================================
# Offline two-path tactile generation check on the A100 cluster.
#
# This script does not train and does not run batch inference. It only compares:
#   Path A: training-validation-style generation from TactileVisualDataset
#   Path B: inference-style generation from load_video_frames()
# on the same 3dprint visual frames and text prompt.
# =============================================================================

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL_ROOT=/defaultShare/models
DATA_ROOT=/defaultShare/data_indoor
CSV_PATH=/18009672469/xy/Show-o/show-o2/contact_indoor_list_tvl.csv
CHECKPOINT=/18009672469/xy/Show-o/show-o2/outputs/showo2-1.5b-tactile-stage-1/checkpoint-18000
VISUAL_DIR=/defaultShare/data_indoor/3dprint/img_gelsight
OUTPUT_DIR=/18009672469/xy/Show-o/show-o2/Inference/path_check_3dprint_ckpt18000

TEXT="The touch of Gadget Holder is smooth, moderate roughness, hardness, akin to flexible rubber, slightly matte yet waxy texture, it is made of rubber."

python check_tactile_generation_paths.py \
    --model_root "${MODEL_ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    --data_root "${DATA_ROOT}" \
    --csv_path "${CSV_PATH}" \
    --visual_dir "${VISUAL_DIR}" \
    --text "${TEXT}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_frames 5 \
    --num_inference_steps 50 \
    --guidance_scale 5.0 \
    --sampling_method euler \
    --time_shifting_factor 3.0 \
    --time_embed_layout auto \
    --seed 42
