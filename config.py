import os

# ===================== PATHS ==================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = r"C:\Users\wajee\PycharmProjects\Derma-Classification\dataset"

TRAIN_IMG_DIR = os.path.join(DATASET_PATH, "images/train")
TEST_IMG_DIR = os.path.join(DATASET_PATH, "images/test")
VAL_IMG_DIR = os.path.join(DATASET_PATH, "images/val")

TRAIN_LABELS_DIR = os.path.join(DATASET_PATH, "labels/ISIC2018_Task3_Training_GroundTruth.csv")
TEST_LABELS_DIR = os.path.join(DATASET_PATH, "labels/ISIC2018_Task3_Test_GroundTruth.csv")
VAL_LABELS_DIR = os.path.join(DATASET_PATH, "labels/ISIC2018_Task3_Validation_GroundTruth.csv")

# ================= GENERAL CONFIG PRE-PROCSSING ===========


run_config = {
    # ==== Dataset loader ====
    "IMAGE_SIZE": 224,
    "NUM_CLASSES": 7,
    "BATCH_SIZE": 16,
    "NUM_WORKERS": 8,

    # ==== CNN model ====
    "MODEL_ARCH": "mobilenetv2",
    "TRAINABLE_LAYERS": 30,
    "PRE_TRAINED": True,

    # ==== Optimizer ====
    "OPTI_NAME": 'adamw',
    "OPTI_LR": 5e-4,
    "OPTI_MOMENTUM": 0.95,
    "LOWER_LR_AFTER": 5,
    "LR_STEP": 5,
    "WEIGHT_DECAY": 0.0005,

    # === Train ===
    "EPOCH": 50,
    "PATIENCE": 7,

}

