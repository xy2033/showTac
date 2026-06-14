#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=offline

# 禁止 HuggingFace Hub 在线下载，强制使用本地文件
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 集群上的本地模型根目录
MODEL_ROOT=/15584226882/Show-o/models

python inference_t2i.py \
  config=configs/showo2_1.5b_demo_432x432.yaml \
  model.showo.pretrained_model_path=${MODEL_ROOT}/show-o2-1.5B \
  model.showo.llm_model_path=${MODEL_ROOT}/Qwen2.5-1.5B-Instruct \
  model.showo.clip_pretrained_model_path=${MODEL_ROOT}/siglip-so400m-patch14-384 \
  dataset.params.validation_prompts_file=/15584226882/Show-o/smoke_prompts.txt \
  batch_size=1 \
  guidance_scale=7.5 \
  num_inference_steps=50