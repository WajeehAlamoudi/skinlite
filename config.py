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
    "BATCH_SIZE": 32,
    "NUM_WORKERS": 8,

    # ==== CNN model ====
    "MODEL_ARCH": "mobilenetv2",
    "TRAINABLE_LAYERS": 70,
    "PRE_TRAINED": False,

    # ==== Optimizer ====
    "OPTI_NAME": 'adam',
    "OPTI_LR": 2e-3,
    "OPTI_MOMENTUM": 0.99,
    "LOWER_LR_AFTER": 7,
    "LR_STEP": 10,
    "WEIGHT_DECAY": 0.0001,

    # === Train ===
    "EPOCH": 50,

}

# callbacks = [EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
#         ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
#         ModelCheckpoint(f"{run_folder}/best_model_trainable30layer{name}.keras", monitor='val_loss', save_best_only=True, verbose=1),
#         CSVLogger(f"{run_folder}/training_log_trainable30layer{name}.csv")]
# metrics=[
#             'accuracy',
#             tf.keras.metrics.Precision(name='precision'),
#             tf.keras.metrics.Recall(name='recall'),
#             tf.keras.metrics.AUC(name='auc'),
#             tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_k_accuracy')
#         ]

# ==== Save ====
# EXPERIMENT_NAME = f"{MODEL_NAME}_{TRAINABLE_LAYERS}layers_{OPTIMIZER}"
# CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, f"best_{EXPERIMENT_NAME}.keras")
# CSV_LOG_PATH = os.path.join(LOGS_DIR, f"training_{EXPERIMENT_NAME}.csv")
