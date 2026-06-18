#!/bin/bash
set -e

# =============================================================================
# Tactile Video Generation — 批量推理脚本 (基于阶段一训练结果)
#
# 对整个测试集进行批量触觉视频生成。
#
# 使用方式:
#   1. 修改 MODEL_ROOT 为集群上模型文件的根目录
#   2. 修改 STAGE1_CHECKPOINT 为阶段一训练的输出路径
#   3. 修改 TACTILE_DATA_ROOT 为触觉数据集根目录
#   4. 修改 TACTILE_CSV_PATH 为触觉数据集 CSV 文件路径
#   5. 修改 OUTPUT_DIR 为输出目录
#   6. bash run_tactile_inference_ssa.sh
#
# 输出:
#   ${OUTPUT_DIR}/
#       {object_name}.mp4      — 生成的触觉视频
#       manifest.jsonl          — 推理记录 (成功/失败, 采样帧号等)
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

# 阶段一训练输出的 checkpoint
STAGE1_CHECKPOINT=outputs/showo2-1.5b-tactile-stage-1_video/checkpoint-48000/unwrapped_model

# ========== 测试集数据路径 ==========
# 触觉数据根目录 (包含每个 object 的子目录，每个子目录下有 img_gelsight/ 和 gelsight/)
TACTILE_DATA_ROOT=/defaultShare/data_indoor

# 触觉数据集 CSV 文件路径 (包含 object_name, contact indices, text_description 等列)
TACTILE_CSV_PATH=contact_indoor_list_tvl.csv

# ========== 输出目录 ==========
OUTPUT_DIR=Inference/test_batch

# ========== Idea 消融超参数 ==========
# 与 run_tactile_stage_one_ssa.sh 保持一致；推理时用于记录/兼容。
# 设为 0 即关闭对应训练方法:
#   VIRTUAL_FORCE_COEFF=0 CONTACT_WEIGHTED_FLOW_ALPHA=0 bash run_tactile_inference_ssa.sh
VIRTUAL_FORCE_COEFF=${VIRTUAL_FORCE_COEFF:-0.1}
CONTACT_WEIGHTED_FLOW_ALPHA=${CONTACT_WEIGHTED_FLOW_ALPHA:-1.0}

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

# ========== 启动批量推理 ==========
python inference_tactile_video.py \
    --batch_test \
    --stage1_checkpoint "${STAGE1_CHECKPOINT}" \
    --vae_path "${MODEL_ROOT}/Wan2.1_VAE.pth" \
    --llm_path "${MODEL_ROOT}/Qwen2.5-1.5B-Instruct" \
    --showo_path "${MODEL_ROOT}/show-o2-1.5B" \
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
    --virtual_force_coeff "${VIRTUAL_FORCE_COEFF}" \
    --contact_weighted_flow_alpha "${CONTACT_WEIGHTED_FLOW_ALPHA}" \
    --fps ${FPS} \
    --save_conditions \
    "${EXTRA_ARGS[@]}"
