import os
import subprocess

# =======================
# 环境变量设置
# =======================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["WANDB_MODE"] = "offline"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# =======================
# 路径配置
# =======================
MODEL_ROOT = "/defaultShare/models"
TACTILE_DATA_ROOT = "/defaultShare/data_indoor"
TACTILE_CSV_PATH = "/Show-o/show-o2/contact_indoor_list_tvl.csv"

# =======================
# Python 和 accelerate 模块路径
# =======================
PYTHON_EXECUTABLE = "/root/miniconda3/envs/showO/bin/python"
ACCELERATE_LAUNCH_MODULE = "/root/miniconda3/envs/showO/lib/python3.10/site-packages/accelerate/commands/launch.py"

# =======================
# accelerate launch 参数
# =======================
args = [
    PYTHON_EXECUTABLE,
    ACCELERATE_LAUNCH_MODULE,
    "train_tactile_stage_one.py",
    "config=configs/showo2_1.5b_tactile_stage_one.yaml",
    f"model.showo.pretrained_model_path={MODEL_ROOT}/show-o2-1.5B",
    f"model.showo.llm_model_path={MODEL_ROOT}/Qwen2.5-1.5B-Instruct",
    f"model.showo.clip_pretrained_model_path={MODEL_ROOT}/siglip-so400m-patch14-384",
    f"model.vae_model.pretrained_model_path={MODEL_ROOT}/Wan2.1_VAE.pth",
    f"dataset.params.tactile_data_root={TACTILE_DATA_ROOT}",
    f"dataset.params.tactile_csv_path={TACTILE_CSV_PATH}",
    "dataset.params.num_frames=5",
    "experiment.generate_model_samples=True",
    "training.batch_size_tactile=1",
    "training.max_train_steps=50000",
    "optimizer.params.learning_rate=0.0001",
]

# =======================
# 启动训练
# =======================
if __name__ == "__main__":
    subprocess.run(args, check=True)