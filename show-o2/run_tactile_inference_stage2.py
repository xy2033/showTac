import os
import subprocess


# =======================
# 环境变量
# =======================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_MODE"] = "offline"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# =======================
# 指定 Python 和 Accelerate
# =======================
PYTHON_EXECUTABLE = "/root/miniconda3/envs/showO/bin/python"
ACCELERATE_LAUNCH_MODULE = "/root/miniconda3/envs/showO/lib/python3.10/site-packages/accelerate/commands/launch.py"

# =======================
# 路径配置
# =======================
MODEL_ROOT = "/defaultShare/models"

STAGE2_CHECKPOINT = (
    "outputs/"
    "showo2-1.5b-tactile-stage-2-qa/"
    "checkpoint-3000/unwrapped_model"
)

TACTILE_DATA_ROOT = "/defaultShare/data_indoor"
TACTILE_CSV_PATH = "contact_indoor_list_tvl.csv"
QA_CSV_PATH = "tac_QA/tactile_qa_pairs.csv"

OUTPUT_DIR = "Inference/stage2_test"


# =======================
# Mode Selection (读取环境变量，若无则使用默认值)
# =======================
QA_MODE = os.getenv("QA_MODE", "true") == "true"


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
VAE_DETERMINISTIC = False


# =======================
# python inference 参数
# =======================
args = [
    PYTHON_EXECUTABLE,
    ACCELERATE_LAUNCH_MODULE,
    
    "inference_tactile_video_stage2.py",

    "--batch_test",
    "--stage2_checkpoint", STAGE2_CHECKPOINT,
    "--vae_path", f"{MODEL_ROOT}/Wan2.1_VAE.pth",
    "--llm_path", f"{MODEL_ROOT}/Qwen2.5-1.5B-Instruct",
    "--showo_path", f"{MODEL_ROOT}/show-o2-1.5B",
    "--siglip_path", f"{MODEL_ROOT}/siglip-so400m-patch14-384",
    "--tactile_data_root", TACTILE_DATA_ROOT,
    "--tactile_csv_path", TACTILE_CSV_PATH,
    "--tactile_qa_csv_path", QA_CSV_PATH,
    "--output_dir", OUTPUT_DIR,
    "--eval_split", EVAL_SPLIT,
    "--num_frames", str(NUM_FRAMES),
    "--num_inference_steps", str(NUM_STEPS),
    "--guidance_scale", str(GUIDANCE_SCALE),
    "--sampling_method", SAMPLING_METHOD,
    "--time_shifting_factor", str(TIME_SHIFTING_FACTOR),
    "--fps", str(FPS),
    "--save_conditions",
]

if QA_MODE:
    args.append("--qa_mode")

if VAE_DETERMINISTIC:
    args.append("--vae_deterministic")


# =======================
# 启动
# =======================
print("[INFO] Mode:", "QA" if QA_MODE else "Pure Generation")
print("[INFO] Running command:")
print(" ".join(args))

subprocess.run(args, check=True)
