import argparse
import torch
import os
from sklearn.metrics import classification_report, f1_score
from torchvision import transforms
from torch.utils.data import DataLoader

# === Local Imports ===
import setting
from capsule.caps_layer import HCapsNet
from capsule.Hcaps_data_loader import HCAPS_ISICDataset
from mobile.mobile_data_loader import ISICDataset
from mobile.mobile_model import MobileNetClassifier

# === Parse CLI ===
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to .pt or .pth file")
parser.add_argument("--model_type", type=str, required=True, choices=["Hcaps", "mobile"], help="Model type")
args = parser.parse_args()

# === Transform ===
test_transform = transforms.Compose([
    transforms.Resize((setting.IMAGE_SIZE, int(setting.IMAGE_SIZE * 1.25))),
    transforms.CenterCrop(setting.IMAGE_SIZE),
    transforms.ToTensor()
])

# === Dataset & Model ===
if args.model_type == "Hcaps":
    test_set = HCAPS_ISICDataset(
        csv_path=setting.TEST_LABELS_DIR,
        img_dir=setting.TEST_IMG_DIR,
        set_state='test',
        transform=test_transform
    )
    model = HCapsNet().to(setting.DEVICE)

elif args.model_type == "mobile":
    test_set = ISICDataset(
        csv_path=setting.TEST_LABELS_DIR,
        img_dir=setting.TEST_IMG_DIR,
        set_state='test',
        transform=test_transform
    )
    model = MobileNetClassifier(num_classes=setting.CLASSES_LEN).to(setting.DEVICE)

test_loader = DataLoader(test_set, batch_size=setting.BATCH_SIZE, shuffle=False, num_workers=setting.NUM_WORKERS)

# === Load Weights ===
if args.model_path.endswith(".pth"):
    model.load_state_dict(torch.load(args.model_path, map_location=setting.DEVICE))
else:  # full .pt model
    if args.model_type == "Hcaps":
        torch.serialization.add_safe_globals([HCapsNet])
    elif args.model_type == "mobile":
        torch.serialization.add_safe_globals([MobileNetClassifier])
    model = torch.load(args.model_path, map_location=setting.DEVICE, weights_only=False)

model.eval()

# === Inference ===
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        if args.model_type == "Hcaps":
            x, _, _, label3 = batch
        else:
            x, label3 = batch

        x, label3 = x.to(setting.DEVICE), label3.to(setting.DEVICE)
        outputs = model(x)

        if args.model_type == "Hcaps":
            preds = outputs["digit3"].norm(dim=-1).argmax(dim=1)
        else:
            preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label3.cpu().numpy())

# === Report ===
print("📊 Classification Report (Fine-Level):")
print(classification_report(all_labels, all_preds, target_names=setting.CLASS_NAMES))
print("Macro F1 Score:", round(f1_score(all_labels, all_preds, average="macro"), 4))



from sklearn.metrics import confusion_matrix
import numpy as np

# === Confusion Matrix ===
cm = confusion_matrix(all_labels, all_preds)
class_names = setting.CLASS_NAMES

print("\n🧩 Confusion Matrix:")
header = "      " + " ".join([f"{name:>6}" for name in class_names])
print(header)
for i, row in enumerate(cm):
    row_str = f"{class_names[i]:<6} " + " ".join([f"{val:>6}" for val in row])
    print(row_str)

