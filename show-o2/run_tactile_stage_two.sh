#!/bin/bash
set -e
# =============================================================================
# Stage 2: Full Model Fine-tuning (QA-NTP Joint Training) — 集群离线版
#
# 使用方式:
#   1. 修改 MODEL_ROOT 为集群上模型文件的根目录
#   2. 修改 TACTILE_DATA_ROOT / TACTILE_CSV_PATH 为实际数据路径
#   3. bash run_tactile_stage_two.sh
#
# Master switch: training.use_tactile_qa (true = QA-NTP, false = pure baseline)
#   use_tactile_qa=true  + ntp_coeff=0.5 → QA-NTP joint training (the idea)
#   use_tactile_qa=false + ntp_coeff=0   → pure Stage2 baseline (no QA mixing)
#
# Stage 2 vs Stage 1:
#   - Stage 1: 冻结 LLM，只训练 fusion/diffusion head，LR=1e-4
#   - Stage 2: 全参数微调，差异化 LR（proj=1e-5, ve/showo=2e-6）
#   - Stage 2 从 Stage 1 的 checkpoint 恢复
# =============================================================================

# ========== GPU 设置 ==========
export CUDA_VISIBLE_DEVICES=0,1

# ========== 离线模式 (集群无网络) ==========
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ========== Configuration Selection ==========
# Default: QA config with QA-NTP enabled
#   CONFIG=configs/showo2_1.5b_tactile_stage_two_qa.yaml (use_tactile_qa=true, ntp_coeff=0.5)
# Pure baseline (no QA mixing at all):
#   USE_TACTILE_QA=false NTP_COEFF=0 bash run_tactile_stage_two.sh
CONFIG=${CONFIG:-configs/showo2_1.5b_tactile_stage_two_qa.yaml}

# Auto-detect: if user picks the non-QA config, default USE_TACTILE_QA to false
if [[ "$CONFIG" != *"qa"* ]] && [[ -z "${USE_TACTILE_QA+x}" ]]; then
    USE_TACTILE_QA=false
fi

# ========== 集群本地路径配置 ==========
# 模型根目录 — 根据你的集群路径修改
MODEL_ROOT=/defaultShare/models

# 触觉数据路径 — 根据你的集群数据路径修改
TACTILE_DATA_ROOT=/defaultShare/data_indoor
TACTILE_CSV_PATH=contact_indoor_list_tvl.csv
QA_CSV_PATH=tac_QA/tactile_qa_pairs.csv

# Stage 1 训练产出（Stage 2 从这里恢复）
STAGE1_CHECKPOINT=outputs/showo2-1.5b-tactile-stage-1_video/checkpoint-final/unwrapped_model

# ========== QA-NTP Idea 消融超参数 ==========
# 设为 false 即关闭 QA（等同于纯 Stage 2 baseline）:
#   USE_TACTILE_QA=false NTP_COEFF=0 bash run_tactile_stage_two.sh
USE_TACTILE_QA=${USE_TACTILE_QA:-true}
NTP_COEFF=${NTP_COEFF:-0.5}

# 设为 false 可强制要求 Stage 1 checkpoint 必须包含 tactile_force_head 权重
ALLOW_MISSING_TACTILE_FORCE_HEAD=${ALLOW_MISSING_TACTILE_FORCE_HEAD:-true}

# ========== 启动训练 ==========
accelerate launch train_tactile_stage_two.py \
    config="${CONFIG}" \
    model.showo.pretrained_model_path="${STAGE1_CHECKPOINT}" \
    model.showo.allow_missing_tactile_force_head="${ALLOW_MISSING_TACTILE_FORCE_HEAD}" \
    model.showo.llm_model_path="${MODEL_ROOT}/Qwen2.5-1.5B-Instruct" \
    model.vae_model.pretrained_model_path="${MODEL_ROOT}/Wan2.1_VAE.pth" \
    dataset.params.tactile_data_root="${TACTILE_DATA_ROOT}" \
    dataset.params.tactile_csv_path="${TACTILE_CSV_PATH}" \
    dataset.params.tactile_qa_csv_path="${QA_CSV_PATH}" \
    dataset.params.num_frames=5 \
    training.use_tactile_qa="${USE_TACTILE_QA}" \
    training.ntp_coeff="${NTP_COEFF}" \
    training.batch_size_tactile=1 \
    training.batch_size_tactile_qa=1 \
    training.max_train_steps=10000 \
    optimizer.params.learning_rate_proj=0.00001 \
    optimizer.params.learning_rate_ve=0.000002 \
    optimizer.params.learning_rate_showo=0.000002
