#!/bin/bash
set -e

# =============================================================================
# Stage 2 Tactile Video Generation & QA — 批量推理脚本
#
# 使用方式:
#   1. 修改 MODEL_ROOT 为集群上模型文件的根目录
#   2. 修改 STAGE2_CHECKPOINT 为 Stage 2 训练输出路径
#   3. 修改 TACTILE_DATA_ROOT / TACTILE_CSV_PATH 为实际数据路径
#   4. 修改 OUTPUT_DIR 为输出目录
#   5. bash run_tactile_inference_stage2.sh
#
# Mode:
#   Pure generation (default): QA_MODE=false → text + visual → tactile video
#   QA mode:                   QA_MODE=true  → question + visual → tactile + answer
#
# 输出:
#   ${OUTPUT_DIR}/
#       {object_name}.mp4       — 生成的触觉视频
#       {object_name}_answer.txt — QA 答案（QA mode）
#       manifest.jsonl           — 推理记录
# =============================================================================

# ========== GPU 设置 ==========
export CUDA_VISIBLE_DEVICES=0

# ========== 离线模式 (集群无网络) ==========
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ========== 集群本地路径配置 ==========
# 模型根目录 — 根据你的集群路径修改
MODEL_ROOT=/defaultShare/models

# Stage 2 训练输出的 checkpoint（修改为你的实际路径）
STAGE2_CHECKPOINT=/Show-o/show-o2/outputs/showo2-1.5b-tactile-stage-2-qa/checkpoint-3000/unwrapped_model

# ========== 测试集数据路径 ==========
TACTILE_DATA_ROOT=/defaultShare/data_indoor
TACTILE_CSV_PATH=/Show-o/show-o2/contact_indoor_list_tvl.csv
QA_CSV_PATH=/Show-o/show-o2/tac_QA/tactile_qa_pairs.csv

# ========== 输出目录 ==========
OUTPUT_DIR=/Show-o/show-o2/Inference/stage2_test

# ========== Mode Selection ==========
# QA_MODE=true  → question + visual → tactile + answer text
# QA_MODE=false → text + visual → tactile video (pure generation)
QA_MODE=${QA_MODE:-true}

# ========== 生成参数 ==========
NUM_FRAMES=5
NUM_STEPS=50
GUIDANCE_SCALE=5.0
SAMPLING_METHOD=euler
TIME_SHIFTING_FACTOR=3.0
FPS=2
EVAL_SPLIT=test
VAE_DETERMINISTIC=false

EXTRA_ARGS=()
if [ "${VAE_DETERMINISTIC}" = "true" ]; then
    EXTRA_ARGS+=(--vae_deterministic)
fi

if [ "${QA_MODE}" = "true" ]; then
    EXTRA_ARGS+=(--qa_mode)
    echo "Mode: QA (question → tactile + answer)"
else
    echo "Mode: Pure Generation (text → tactile)"
fi

# ========== 启动批量推理 ==========
python inference_tactile_video_stage2.py \
    --batch_test \
    --stage2_checkpoint "${STAGE2_CHECKPOINT}" \
    --vae_path "${MODEL_ROOT}/Wan2.1_VAE.pth" \
    --llm_path "${MODEL_ROOT}/Qwen2.5-1.5B-Instruct" \
    --showo_path "${MODEL_ROOT}/show-o2-1.5B" \
    --siglip_path "${MODEL_ROOT}/siglip-so400m-patch14-384" \
    --tactile_data_root "${TACTILE_DATA_ROOT}" \
    --tactile_csv_path "${TACTILE_CSV_PATH}" \
    --tactile_qa_csv_path "${QA_CSV_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --eval_split "${EVAL_SPLIT}" \
    --num_frames ${NUM_FRAMES} \
    --num_inference_steps ${NUM_STEPS} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --sampling_method "${SAMPLING_METHOD}" \
    --time_shifting_factor ${TIME_SHIFTING_FACTOR} \
    --fps ${FPS} \
    --save_conditions \
    "${EXTRA_ARGS[@]}"
