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
    "NUM_WORKERS": 0,

    # ==== CNN model ====
    "MODEL_ARCH": "mobilenetv2",  # efficientnet_b0, shufflenet_v2_x1_0
    "TRAINABLE_LAYERS": 7,
    "PRE_TRAINED": True,
    "PRIMARY_UNITS": 32,
    "PRIMARY_UNIT_SIZE": 8,

    # ==== Optimizer ====
    "OPTI_NAME": 'adamw',  # adam, sgd, rmsprop, nadam
    "OPTI_LR": 2e-3,
    "OPTI_MOMENTUM": 0.95,
    "LOWER_LR_AFTER": 10,
    "LR_STEP": 5,
    "WEIGHT_DECAY": 1e-2,

    # === Train ===
    "EPOCH": 1,
    "PATIENCE": 1,
    "MIX_UP": True,
    "MIXUP_ALPHA": 0.2,

    # == Loss fun ==
    "LOSS_FUN": 'capsule_margin',  # focal, cross_entropy, class_weight
    "LOSS_GAMMA": None,  # not use with class_weight, cross_entropy
    "LOSS_ALPHA": None,  # torch.tensor([0.2, 0.03, 0.25, 0.35, 0.2, 0.9, 1.0]) not use with class_weight, cross_entropy
    "LOSS_REDUCTION": 'mean',  # sum

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
