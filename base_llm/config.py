import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# ============================================================
#                    模型和 LoRA 配置
# ============================================================

# ---------- 模型路径 ----------
MODEL_PATH = os.path.join(BASE_DIR, "models", "bert-base-uncased")

# ---------- LoRA 参数 ----------
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["query", "value"]

# ============================================================
#                    训练参数
# ============================================================

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
LOCAL_EPOCHS = 1
ROUNDS = 3
NUM_CLIENTS = 10
TEST_SPLIT = 0.2
MAX_LEN = 64

CLIP_NORM = 10.0
WEIGHT_DECAY = 1e-4

# 实验一配置

# 数据集: sst2, mnli, qnli
DATASET_NAME = "qnli"

# 模型名称（用于记录和画图）
MODEL_NAME = "bert-base"

# 聚合方法: bfsvd, full, svd, freeze_a, freeze_b
AGG_METHOD = "full"

# 实验一关闭加密和防御
USE_ENCRYPTION = False
USE_FULD = False
USE_MASK = False

# 实验记录目录
EXP1_RESULT_DIR = os.path.join(RESULT_DIR, "exp1")
os.makedirs(EXP1_RESULT_DIR, exist_ok=True)
EXP1_RUN_TAG = "quick"


# 旧版聚合模式（保留兼容，实验一不使用）
AGGREGATION_MODE = "fedsvd"
SERIALIZE_CLIENTS = False