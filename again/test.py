import torch
import argparse
import os
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms
from capsule.caps_layer import *
from capsule.Hcaps_data_loader import *
from mobile.mobile_data_loader import *
from mobile.mobile_model import *
from transforms import *
from torch.utils.data import DataLoader
from utils import *

# === Parse arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to .pth model weights")
parser.add_argument("--model_type", type=str, required=True, choices=["Hcaps", "mobile"], help="Model architecture")
args = parser.parse_args()

# === Transforms ===
test_transform = transforms.Compose([
    transforms.Resize((setting.IMAGE_SIZE, setting.IMAGE_SIZE)),
    transforms.ToTensor()
])

# === Load Dataset ===
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

test_loader = DataLoader(test_set, batch_size=setting.BATCH_SIZE, shuffle=False, num_workers=2)

# === Load Weights ===
model.load_state_dict(torch.load(args.model_path, map_location=setting.DEVICE))
model.eval()

# === Inference ===
all_preds, all_labels = [], []

with torch.no_grad():
    for data in test_loader:
        if args.model_type == "Hcaps":
            x, _, _, label3 = data
        else:  # mobile
            x, label3 = data

        x = x.to(setting.DEVICE)
        label3 = label3.to(setting.DEVICE)

        outputs = model(x)

        if args.model_type == "Hcaps":
            pred = outputs["digit3"].norm(dim=-1).argmax(dim=1)
        else:
            pred = outputs.argmax(dim=1)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(label3.cpu().numpy())

# === Report ===
print("📊 Classification Report (Fine-Level):")
print(classification_report(all_labels, all_preds, target_names=setting.CLASS_NAMES))
print("Macro F1 Score:", round(f1_score(all_labels, all_preds, average="macro"), 4))
