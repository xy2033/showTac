#!/root/miniconda3/envs/showO/bin/python
import os
import subprocess


# =======================
# 指定 Python 和 Accelerate
# =======================
PYTHON_EXECUTABLE = "/root/miniconda3/envs/showO/bin/python"
ACCELERATE_LAUNCH_MODULE = "/root/miniconda3/envs/showO/lib/python3.10/site-packages/accelerate/commands/launch.py"


# =======================
# 环境变量
# =======================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["WANDB_MODE"] = "offline"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# =======================
# 路径配置
# =======================
MODEL_ROOT = "/defaultShare/models"

STAGE1_CHECKPOINT = (
    "/Show-o/show-o2/outputs/"
    "showo2-1.5b-tactile-stage-1_video/"
    "checkpoint-45000/unwrapped_model"
)

TACTILE_DATA_ROOT = "/defaultShare/data_indoor"
TACTILE_CSV_PATH = "/Show-o/show-o2/contact_indoor_list_tvl.csv"

OUTPUT_DIR = "/Show-o/show-o2/Inference/test_batch"


# =======================
# 生成参数
# =======================
NUM_FRAMES = 5
NUM_STEPS = 50
GUIDANCE_SCALE = 5.0
SAMPLING_METHOD = "euler"
TIME_SHIFTING_FACTOR = 3.0
FPS = 2
EVAL_SPLIT = "test"
VAE_DETERMINISTIC = True


# =======================
# accelerate launch 参数
# =======================
args = [
    PYTHON_EXECUTABLE,
    ACCELERATE_LAUNCH_MODULE,

    # accelerate 参数
    "--num_processes", "1",
    "--num_machines", "1",

    # 被启动的推理脚本
    "inference_tactile_video.py",

    # 推理参数
    "--batch_test",
    "--stage1_checkpoint", STAGE1_CHECKPOINT,
    "--vae_path", f"{MODEL_ROOT}/Wan2.1_VAE.pth",
    "--llm_path", f"{MODEL_ROOT}/Qwen2.5-1.5B-Instruct",
    "--siglip_path", f"{MODEL_ROOT}/siglip-so400m-patch14-384",
    "--tactile_data_root", TACTILE_DATA_ROOT,
    "--tactile_csv_path", TACTILE_CSV_PATH,
    "--output_dir", OUTPUT_DIR,
    "--eval_split", EVAL_SPLIT,
    "--num_frames", str(NUM_FRAMES),
    "--num_inference_steps", str(NUM_STEPS),
    "--guidance_scale", str(GUIDANCE_SCALE),
    "--sampling_method", SAMPLING_METHOD,
    "--time_shifting_factor", str(TIME_SHIFTING_FACTOR),
    "--time_embed_layout", "auto",
    "--fps", str(FPS),
    "--save_conditions",
    "--showo_path", f"{MODEL_ROOT}/show-o2-1.5B",
]

if VAE_DETERMINISTIC:
    args.append("--vae_deterministic")


# =======================
# 启动
# =======================
print("[INFO] Running command:")
print(" ".join(args))

subprocess.run(args, check=True)