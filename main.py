import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# === Paths ===
# DATASET_PATH = r"C:\Users\wajee\PycharmProjects\Derma-Classification\dataset"
DATASET_PATH = "/content/drive/MyDrive/dataset"
TRAIN_IMG_DIR = os.path.join(DATASET_PATH, "images/train")
VAL_IMG_DIR = os.path.join(DATASET_PATH, "images/val")
TRAIN_LABELS_PATH = os.path.join(DATASET_PATH, "labels/ISIC2018_Task3_Training_GroundTruth.csv")
VAL_LABELS_PATH = os.path.join(DATASET_PATH, "labels/ISIC2018_Task3_Validation_GroundTruth.csv")

# === Label Mapping ===
CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}


# === Custom Dataset ===
class ISICDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        for _, row in self.data.iterrows():
            image_id = row['image']
            label = row[CLASS_NAMES].astype(int).idxmax()
            self.samples.append((f"{image_id}.jpg", LABEL_MAP[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# === Capsule Components ===
class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=512, out_channels=32, kernel_size=7, stride=2):
        super().__init__()
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride)
            for _ in range(num_capsules)
        ])

    def forward(self, x):
        B = x.size(0)
        u = [caps(x).view(B, -1, 1) for caps in self.capsules]
        u = torch.cat(u, dim=-1)
        return self.squash(u)

    def squash(self, x, dim=-1):
        norm = torch.norm(x, dim=dim, keepdim=True)
        scale = (norm ** 2) / (1 + norm ** 2)
        return scale * x / (norm + 1e-7) * 3.0  # 🚀 Boost final vector


class DigitCaps(nn.Module):
    def __init__(self, input_caps, input_dim, num_classes, output_dim=16, routing_iters=3):
        super().__init__()
        W = torch.empty(1, input_caps, num_classes, output_dim, input_dim)
        nn.init.xavier_uniform_(W)  # 🧠 better distribution
        self.W = nn.Parameter(W)
        self.routing_iters = routing_iters
        self.num_classes = num_classes

    def forward(self, x):
        B = x.size(0)
        x = x.unsqueeze(2).unsqueeze(4)
        W = self.W.expand(B, -1, -1, -1, -1)
        u_hat = torch.matmul(W, x).squeeze(-1)
        b_ij = torch.zeros(B, x.size(1), self.num_classes, 1, device=x.device)
        for _ in range(self.routing_iters):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=1)
            v_j = self.squash(s_j)
            if _ < self.routing_iters - 1:
                b_ij = b_ij + (u_hat * v_j.unsqueeze(1)).sum(-1, keepdim=True)
        print("DigitCaps norms:", torch.norm(v_j, dim=-1).mean().item())
        return v_j

    def squash(self, x, dim=-1):
        norm = torch.norm(x, dim=dim, keepdim=True)
        scale = (norm ** 2) / (1 + norm ** 2)
        return scale * x / (norm + 1e-7) * 3.0  # 🚀 Boost final vector


class CapsuleNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.backbone = nn.Sequential(*list(mobilenet.features.children()))
        self.adapter = nn.Conv2d(1280, 512, kernel_size=1)
        self.primary_caps = PrimaryCaps(in_channels=512)

        # Dynamically determine the number of input capsules
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_feat = self.adapter(self.backbone(dummy_input))
            dummy_caps = self.primary_caps(dummy_feat)
            input_caps = dummy_caps.shape[1]  # ← actual value (e.g., 1152)

        self.digit_caps = DigitCaps(input_caps=input_caps, input_dim=8, num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.adapter(x)
        x = F.relu(x)
        x = self.primary_caps(x)
        print("PrimaryCaps output shape:", x.shape)
        x = self.digit_caps(x)
        return x.norm(dim=-1)


class CapsuleMarginLoss(nn.Module):
    def __init__(self, margin=0.4, downweight=0.5):
        super().__init__()
        self.margin = margin
        self.downweight = downweight

    def forward(self, pred, labels):
        # pred: [B, 7], labels: [B] (integer indices)
        one_hot = F.one_hot(labels, num_classes=pred.size(1)).float()

        # Margin loss
        left = F.relu(self.margin - pred) ** 2
        right = F.relu(pred - (1 - self.margin)) ** 2

        # Apply loss
        loss = one_hot * left + self.downweight * (1 - one_hot) * right
        loss = loss.sum(dim=1).mean()  # sum across classes, mean across batch

        # Debug once every few steps
        if torch.rand(1).item() < 0.1:
            print("Mean pred norm:", pred.norm(dim=1).mean().item())
            print("Sample true class scores:", pred[range(len(labels)), labels].detach().cpu().tolist())
            print("Sample total loss:", loss.item())

        return loss


# === Config ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

train_set = ISICDataset(TRAIN_LABELS_PATH, TRAIN_IMG_DIR, transform=transform)
val_set = ISICDataset(VAL_LABELS_PATH, VAL_IMG_DIR, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

model = CapsuleNet().to(device)
criterion = CapsuleMarginLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# === Training ===
train_losses = []
val_losses = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        print("Output stats:", outputs.min().item(), outputs.max().item())
        print("Labels (raw indices):", labels[:8].tolist())
        print("Label example:", labels[0].item())
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Optional: Evaluate on validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title("CapsuleNet Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
