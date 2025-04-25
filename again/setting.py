import os

import torch

DATASET_PATH = r"C:\Users\wajee\PycharmProjects\Derma-Classification\dataset"

TRAIN_IMG_DIR = os.path.join(DATASET_PATH, "images/train")
TEST_IMG_DIR = os.path.join(DATASET_PATH, "images/test")
VAL_IMG_DIR = os.path.join(DATASET_PATH, "images/val")

TRAIN_LABELS_DIR = os.path.join(DATASET_PATH, "labels/ISIC2018_Task3_Training_GroundTruth.csv")
TEST_LABELS_DIR = os.path.join(DATASET_PATH, "labels/ISIC2018_Task3_Test_GroundTruth.csv")
VAL_LABELS_DIR = os.path.join(DATASET_PATH, "labels/ISIC2018_Task3_Validation_GroundTruth.csv")

# ===================
CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
CLASS_MULTIPLIER = {
    6: 0,  # VASC
    5: 0,  # DF
    4: 0,  # BKL
    3: 0,  # AKIEC
    2: 0,  # BCC
    1: 0,  # NV (downsample)
    0: 0  # MEL
}
model = 'Hcaps'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 32
TRAIN_SHUFFLE = True
NUM_WORKERS = 8
CLASSES_LEN = len(CLASS_NAMES)
EPOCHS = 20
H_CLASSES_COUNT = [2, 3, 7]
LAMBDA_RECON = 0.0005
LEARNING_RATE = 1e-3

# ===================




