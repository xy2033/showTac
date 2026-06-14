#!/bin/bash
set -e

# =============================================================================
# Tactile Video Generation — 批量推理脚本
# =============================================================================

# ========== 指定 Python 和 Accelerate ==========
PYTHON_EXECUTABLE="/root/miniconda3/envs/showO/bin/python"
ACCELERATE_LAUNCH_MODULE="/root/miniconda3/envs/showO/lib/python3.10/site-packages/accelerate/commands/launch.py"

# ========== GPU 设置 ==========
export CUDA_VISIBLE_DEVICES=0

# ========== 离线模式 ==========
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ========== 集群本地路径配置 ==========
MODEL_ROOT=/defaultShare/models

# 阶段一训练输出的 checkpoint
STAGE1_CHECKPOINT=/Show-o/show-o2/outputs/showo2-1.5b-tactile-stage-1_video/checkpoint-45000/unwrapped_model

# ========== 测试集数据路径 ==========
TACTILE_DATA_ROOT=/defaultShare/data_indoor
TACTILE_CSV_PATH=/Show-o/show-o2/contact_indoor_list_tvl.csv

# ========== 输出目录 ==========
OUTPUT_DIR=/Show-o/show-o2/Inference/test_batch

# ========== 生成参数 ==========
NUM_FRAMES=5
NUM_STEPS=50
GUIDANCE_SCALE=5.0
SAMPLING_METHOD=euler
TIME_SHIFTING_FACTOR=3.0
FPS=2
EVAL_SPLIT=test
VAE_DETERMINISTIC=true

EXTRA_ARGS=()
if [ "${VAE_DETERMINISTIC}" = "true" ]; then
    EXTRA_ARGS+=(--vae_deterministic)
fi

# ========== 启动批量推理 ==========
"${PYTHON_EXECUTABLE}" "${ACCELERATE_LAUNCH_MODULE}" \
    --num_processes 1 \
    --num_machines 1 \
    inference_tactile_video.py \
    --batch_test \
    --stage1_checkpoint "${STAGE1_CHECKPOINT}" \
    --vae_path "${MODEL_ROOT}/Wan2.1_VAE.pth" \
    --llm_path "${MODEL_ROOT}/Qwen2.5-1.5B-Instruct" \
    --siglip_path "${MODEL_ROOT}/siglip-so400m-patch14-384" \
    --tactile_data_root "${TACTILE_DATA_ROOT}" \
    --tactile_csv_path "${TACTILE_CSV_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --eval_split "${EVAL_SPLIT}" \
    --num_frames ${NUM_FRAMES} \
    --num_inference_steps ${NUM_STEPS} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --sampling_method "${SAMPLING_METHOD}" \
    --time_shifting_factor ${TIME_SHIFTING_FACTOR} \
    --time_embed_layout auto \
    --fps ${FPS} \
    --save_conditions \
    --showo_path "${MODEL_ROOT}/show-o2-1.5B" \
    "${EXTRA_ARGS[@]}"