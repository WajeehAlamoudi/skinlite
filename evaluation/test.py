import os
import torch
import yaml
from sklearn.metrics import classification_report
from models.model import build_model
from data.isic_loader import ISICDataset

# 🔸 Set your run directory here
test_dir = r"C:\Users\wajee\Downloads\skinlite_runs_backup-20250418T150559Z-001\skinlite_runs_backup\run_20250418_1427"

# 🔹 Load config from YAML
with open(os.path.join(test_dir, "config.yaml"), "r") as f:
    test_config = yaml.safe_load(f)
print(test_config)
# 🔹 Load test dataset
test_dataset = ISICDataset(set_state='test', output_size=test_config["IMAGE_SIZE"])
test_loader = test_dataset.get_loader(batch_size=1, num_workers=0)

# 🔹 Build model
model = build_model(
    arch=test_config["MODEL_ARCH"],
    input_size=test_config["IMAGE_SIZE"],
    num_classes=test_config["NUM_CLASSES"],
    trainable_layers=test_config["TRAINABLE_LAYERS"],
    pretrained=False
)

# 🔹 Load weights
model.load_state_dict(torch.load(os.path.join(test_dir, "best_model.pth"), map_location="cpu"))
model.eval()

# 🔹 Predict
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

# 🔹 Classification Report
target_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))
