import os

import torch

# ===================== PATHS ==================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = r"C:\Users\wajee\PycharmProjects\Derma-Classification\dataset"

# BASE_DIR = "/content/skinlite"
#
# DATASET_PATH = "/content/drive/MyDrive/dataset"

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
    "MODEL_ARCH": "mobilenetv2",  # efficientnet_b0, shufflenet_v2_x1_0
    "TRAINABLE_LAYERS": 7,
    "PRE_TRAINED": True,

    # ==== Optimizer ====
    "OPTI_NAME": 'adamw',  # adam, sgd, rmsprop, nadam
    "OPTI_LR": 1e-5,
    "OPTI_MOMENTUM": 0.95,
    "LOWER_LR_AFTER": 10,
    "LR_STEP": 5,
    "WEIGHT_DECAY": 0.001,

    # === Train ===
    "EPOCH": 100,
    "PATIENCE": 12,
    "LOSS_FUN": 'class_weight',  # focal, cross_entropy
    "LOSS_GAMMA": 2,    # not use with class_weight
    "LOSS_ALPHA": torch.tensor([0.2, 0.03, 0.25, 0.35, 0.2, 0.9, 1.0]),  # not use with class_weight
    "LOSS_REDUCTION": 'mean',  # sum
    "MIX_UP": True,
    "MIXUP_ALPHA": 0.2,

}
""""
isic_loader.py

model.py

optimizer.py

train.py

helper.py

loss_functions.py

transforms.py

config.py
"""
