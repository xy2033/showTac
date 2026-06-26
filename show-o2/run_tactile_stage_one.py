import os
import subprocess

# =======================
# 环境变量设置
# =======================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["WANDB_MODE"] = "offline"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# =======================
# 路径配置
# =======================
MODEL_ROOT = "/defaultShare/models"
TACTILE_DATA_ROOT = "/defaultShare/data_indoor"
TACTILE_CSV_PATH = "contact_indoor_list_tvl.csv"

# =======================
# Python 和 accelerate 模块路径
# =======================
PYTHON_EXECUTABLE = "/root/miniconda3/envs/showO/bin/python"
ACCELERATE_LAUNCH_MODULE = "/root/miniconda3/envs/showO/lib/python3.10/site-packages/accelerate/commands/launch.py"
# =======================
# Motion aux 消融超参数 (读取环境变量，若无则使用默认值)
# v3: 运动图训练时 on-the-fly 计算，无 cache
# =======================
MOTION_AUX_ENABLED = os.getenv("MOTION_AUX_ENABLED", "true")
MOTION_MODE = os.getenv("MOTION_MODE", "motion_condition")
MOTION_AUX_COEFF = os.getenv("MOTION_AUX_COEFF", "0.1")
MOTION_USE_CONTACT_MASK = os.getenv("MOTION_USE_CONTACT_MASK", "true")
MOTION_CONTACT_THRESHOLD_MODE = os.getenv("MOTION_CONTACT_THRESHOLD_MODE", "window_percentile")
MOTION_CONTACT_PERCENTILE = os.getenv("MOTION_CONTACT_PERCENTILE", "96.0")
MOTION_CONTACT_K = os.getenv("MOTION_CONTACT_K", "3.0")
MOTION_MAX = os.getenv("MOTION_MAX", "32.0")
MOTION_GAMMA = os.getenv("MOTION_GAMMA", "0.5")
MAX_TRAIN_STEPS = os.getenv("MAX_TRAIN_STEPS", "50000")
TRAIN_OUTPUT_DIR = os.getenv("TRAIN_OUTPUT_DIR", "outputs/showo2-1.5b-tactile-stage-1_motion")
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
    f"experiment.output_dir={TRAIN_OUTPUT_DIR}",
    f"dataset.params.tactile_data_root={TACTILE_DATA_ROOT}",
    f"dataset.params.tactile_csv_path={TACTILE_CSV_PATH}",
    "dataset.params.num_frames=5",
    "experiment.generate_model_samples=True",
    "training.batch_size_tactile=2",
    f"training.max_train_steps={MAX_TRAIN_STEPS}",
    f"motion_aux.enabled={MOTION_AUX_ENABLED}",
    f"motion_aux.mode={MOTION_MODE}",
    f"motion_aux.coeff={MOTION_AUX_COEFF}",
    f"motion_aux.use_contact_mask={MOTION_USE_CONTACT_MASK}",
    f"motion_aux.contact_threshold_mode={MOTION_CONTACT_THRESHOLD_MODE}",
    f"motion_aux.contact_percentile={MOTION_CONTACT_PERCENTILE}",
    f"motion_aux.contact_k={MOTION_CONTACT_K}",
    f"motion_aux.motion_max={MOTION_MAX}",
    f"motion_aux.motion_gamma={MOTION_GAMMA}",
    "optimizer.params.learning_rate=0.0001",
]

# =======================
# 启动训练
# =======================
if __name__ == "__main__":
    subprocess.run(args, check=True)
