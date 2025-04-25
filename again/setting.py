import os
import torch


DATASET_PATH = "/content/drive/MyDrive/dataset"
TRAIN_IMG_DIR = os.path.join(DATASET_PATH, "images/train")
TEST_IMG_DIR = os.path.join(DATASET_PATH, "images/test")
VAL_IMG_DIR = os.path.join(DATASET_PATH, "images/val")

TRAIN_LABELS_DIR = os.path.join(DATASET_PATH, "labels/ISIC2018_Task3_Training_GroundTruth.csv")
TEST_LABELS_DIR = os.path.join(DATASET_PATH, "labels/ISIC2018_Task3_Test_GroundTruth.csv")
VAL_LABELS_DIR = os.path.join(DATASET_PATH, "labels/ISIC2018_Task3_Validation_GroundTruth.csv")

# ===================
CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
CLASS_MULTIPLIER = {
    6: 5,  # VASC
    5: 5,  # DF
    4: 0,  # BKL
    3: 3,  # AKIEC
    2: 2,  # BCC
    1: -2,  # NV (downsample)
    0: 3  # MEL
}
model = 'mobile' # , Hcaps
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 32
TRAIN_SHUFFLE = True
NUM_WORKERS = 8
TRAINABLE = 7
CLASSES_LEN = len(CLASS_NAMES)
EPOCHS = 5
H_CLASSES_COUNT = [2, 3, 7]
LAMBDA_RECON = 0.0005
LEARNING_RATE = 1e-3

# ===================




