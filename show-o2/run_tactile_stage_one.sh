#!/bin/bash
set -e

# =============================================================================
# Stage 1: Tactile Domain Adaptation (集群离线版)
#
# 使用方式:
#   1. 修改 MODEL_ROOT 为集群上模型文件的根目录
#   2. 修改 TACTILE_DATA_ROOT / TACTILE_CSV_PATH 为实际数据路径
#   3. bash run_tactile_stage_one.sh
#
# 训练完成后:
#   模型保存在 outputs/showo2-1.5b-tactile-stage-1/checkpoint-final/
#   可直接用于推理: bash run_tactile_inference.sh
# =============================================================================
# ========== 指定 Python 和 Accelerate ==========
PYTHON_EXECUTABLE="/root/miniconda3/envs/showO/bin/python"
ACCELERATE_LAUNCH_MODULE="/root/miniconda3/envs/showO/lib/python3.10/site-packages/accelerate/commands/launch.py"
# ========== GPU 设置 ==========
export CUDA_VISIBLE_DEVICES=0,1

# ========== 离线模式 (集群无网络) ==========
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ========== 集群本地路径配置 ==========
# 模型根目录 — 根据你的集群路径修改
MODEL_ROOT=/defaultShare/models

# 触觉数据路径 — 根据你的集群数据路径修改
TACTILE_DATA_ROOT=/defaultShare/data_indoor
TACTILE_CSV_PATH=contact_indoor_list_tvl.csv
# ========== Idea 消融超参数 ==========
# 设为 0 即关闭对应方法:
#   VIRTUAL_FORCE_COEFF=0 CONTACT_WEIGHTED_FLOW_ALPHA=0 bash run_tactile_stage_one_ssa.sh
VIRTUAL_FORCE_COEFF=${VIRTUAL_FORCE_COEFF:-0.1}
# CONTACT_WEIGHTED_FLOW_ALPHA=${CONTACT_WEIGHTED_FLOW_ALPHA:-1.0}
CONTACT_WEIGHTED_FLOW_ALPHA=0

# ========== 启动训练 ==========
"${PYTHON_EXECUTABLE}" "${ACCELERATE_LAUNCH_MODULE}" train_tactile_stage_one.py \
    config=configs/showo2_1.5b_tactile_stage_one.yaml \
    model.showo.pretrained_model_path="${MODEL_ROOT}/show-o2-1.5B" \
    model.showo.llm_model_path="${MODEL_ROOT}/Qwen2.5-1.5B-Instruct" \
    model.showo.clip_pretrained_model_path="${MODEL_ROOT}/siglip-so400m-patch14-384" \
    model.vae_model.pretrained_model_path="${MODEL_ROOT}/Wan2.1_VAE.pth" \
    dataset.params.tactile_data_root="${TACTILE_DATA_ROOT}" \
    dataset.params.tactile_csv_path="${TACTILE_CSV_PATH}" \
    dataset.params.num_frames=5 \
    experiment.generate_model_samples=True \
    training.batch_size_tactile=1 \
    training.max_train_steps=50000 \
    training.virtual_force_coeff="${VIRTUAL_FORCE_COEFF}" \
    training.contact_weighted_flow_alpha="${CONTACT_WEIGHTED_FLOW_ALPHA}" \
    optimizer.params.learning_rate=0.0001
