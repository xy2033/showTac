import os
import subprocess

# =======================
# 环境变量设置
# =======================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_MODE"] = "offline"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# =======================
# 路径配置
# =======================
MODEL_ROOT = "/defaultShare/models"
TACTILE_DATA_ROOT = "/defaultShare/data_indoor"
TACTILE_CSV_PATH = "contact_indoor_list_tvl.csv"
QA_CSV_PATH = "tac_QA/tactile_qa_pairs.csv"
STAGE1_CHECKPOINT = "outputs/showo2-1.5b-tactile-stage-1_video/checkpoint-final/unwrapped_model"

# =======================
# Python 和 accelerate 模块路径
# =======================
PYTHON_EXECUTABLE = "/root/miniconda3/envs/showO/bin/python"
ACCELERATE_LAUNCH_MODULE = "/root/miniconda3/envs/showO/lib/python3.10/site-packages/accelerate/commands/launch.py"

# =======================
# QA-NTP Idea 消融超参数 (读取环境变量，若无则使用默认值)
# =======================
USE_TACTILE_QA = os.getenv("USE_TACTILE_QA", "true")
NTP_COEFF = os.getenv("NTP_COEFF", "0.5")
ALLOW_MISSING_TACTILE_FORCE_HEAD = os.getenv("ALLOW_MISSING_TACTILE_FORCE_HEAD", "true")

# =======================
# accelerate launch 参数
# =======================
args = [
    PYTHON_EXECUTABLE,
    ACCELERATE_LAUNCH_MODULE,
    "train_tactile_stage_two.py",
    "config=configs/showo2_1.5b_tactile_stage_two_qa.yaml",
    f"model.showo.pretrained_model_path={STAGE1_CHECKPOINT}",
    f"model.showo.allow_missing_tactile_force_head={ALLOW_MISSING_TACTILE_FORCE_HEAD}",
    f"model.showo.llm_model_path={MODEL_ROOT}/Qwen2.5-1.5B-Instruct",
    f"model.vae_model.pretrained_model_path={MODEL_ROOT}/Wan2.1_VAE.pth",
    f"dataset.params.tactile_data_root={TACTILE_DATA_ROOT}",
    f"dataset.params.tactile_csv_path={TACTILE_CSV_PATH}",
    f"dataset.params.tactile_qa_csv_path={QA_CSV_PATH}",
    "dataset.params.num_frames=5",
    f"training.use_tactile_qa={USE_TACTILE_QA}",
    f"training.ntp_coeff={NTP_COEFF}",
    "training.batch_size_tactile=1",
    "training.batch_size_tactile_qa=1",
    "training.max_train_steps=10000",
    "optimizer.params.learning_rate_proj=0.00001",
    "optimizer.params.learning_rate_ve=0.000002",
    "optimizer.params.learning_rate_showo=0.000002",
]

# =======================
# 启动训练
# =======================
if __name__ == "__main__":
    subprocess.run(args, check=True)
