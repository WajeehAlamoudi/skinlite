import os

import torch

# ===================== PATHS ==================

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#
# DATASET_PATH = r"C:\Users\wajee\PycharmProjects\Derma-Classification\dataset"

BASE_DIR = "/content/skinlite"

DATASET_PATH = "/content/drive/MyDrive/dataset"

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
    "BATCH_SIZE": 32,
    "NUM_WORKERS": 8,

    # ==== CNN model ====
    "MODEL_ARCH": "mobilenetv2",    # efficientnet_b0, shufflenet_v2_x1_0
    "TRAINABLE_LAYERS": 7,
    "PRE_TRAINED": True,

    # ==== Optimizer ====
    "OPTI_NAME": 'adamw',   # adam, sgd, rmsprop, nadam
    "OPTI_LR": 5e-4,
    "OPTI_MOMENTUM": 0.95,
    "LOWER_LR_AFTER": 5,
    "LR_STEP": 2,
    "WEIGHT_DECAY": 0.0005,

    # === Train ===
    "EPOCH": 50,
    "PATIENCE": 7,
    "LOSS_FUN": 'focal',  # class_weight
    "LOSS_GAMMA": 2,
    "LOSS_ALPHA": torch.tensor([0.2, 0.03, 0.25, 0.35, 0.2, 0.9, 1.0]),
    "LOSS_REDUCTION": 'mean',   # sum

}
