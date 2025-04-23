import csv

import torch

import config
from data.isic_loader import ISICDataset
import os
from utils.helpers import setup_run_folder, compute_class_weights_from_labels
from utils.loss_functions import CapsuleLoss
from models.model import build_model
from models.optimizer import get_optimizer
import torch.nn.functional as F

# 🔸 Step 1: Create a run folder
run_dir, log_csv_path = setup_run_folder(config.BASE_DIR, config.run_config)

# 🔸 Step 2: Load dataset
train_dataset = ISICDataset(set_state='train', output_size=config.run_config['IMAGE_SIZE'])
val_dataset = ISICDataset(set_state='val', output_size=config.run_config['IMAGE_SIZE'])
train_loader = train_dataset.get_loader(
    batch_size=config.run_config['BATCH_SIZE'],
    num_workers=config.run_config['NUM_WORKERS']
)
val_loader = val_dataset.get_loader(
    batch_size=config.run_config['BATCH_SIZE'],
    num_workers=config.run_config['NUM_WORKERS']
)

# 🔸 Step 3: Initialize model
model = build_model(
    arch=config.run_config['MODEL_ARCH'],
    num_classes=config.run_config['NUM_CLASSES'],
    trainable_layers=config.run_config['TRAINABLE_LAYERS'],
    pretrained=config.run_config['PRE_TRAINED'],
    num_primary_units=config.run_config['PRIMARY_UNITS'],
    primary_unit_size=config.run_config['PRIMARY_UNIT_SIZE']
)

# 🔸 Step 4: Optimizer and Scheduler
optimizer, scheduler = get_optimizer(
    optim_params=model.parameters(),
    optim_name=config.run_config['OPTI_NAME'],
    optim_lr=config.run_config['OPTI_LR'],
    optim_momentum=config.run_config['OPTI_MOMENTUM'],
    change_after=config.run_config['LOWER_LR_AFTER'],
    lr_step=config.run_config['LR_STEP'],
    weight_decay=config.run_config['WEIGHT_DECAY']
)

# 🔸 Step 5: Setup device and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = CapsuleLoss(
    pos_class=config.run_config['LOSS_+_class'],
    neg_class=config.run_config['LOSS_-_class'],
    penalty=config.run_config['LOSS_PENALTY'],
    reduction=config.run_config['LOSS_REDUCTION'],
    class_weights=compute_class_weights_from_labels(train_dataset.label_paths, config.run_config['NUM_CLASSES'])
)

# 5.1 retrieve train config
num_epochs = config.run_config['EPOCH']
patience = config.run_config['PATIENCE']

best_val_accuracy = 0.0
patience_counter = 0

# 🔸 Step 6: Train
print("✅ Training setup complete. Beginning training...")
for epoch in range(num_epochs):
    print(f"\n🔁 Epoch {epoch + 1}/{num_epochs}")

    # === Train ===
    model.train()
    train_loss, train_correct = 0.0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        caps_output, reconstructions, pred_y = model(images)
        one_hot_labels = F.one_hot(labels, num_classes=config.run_config['NUM_CLASSES']).float()
        loss = criterion(caps_output, one_hot_labels, images, reconstructions)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        # train_correct += (outputs.argmax(1) == labels).sum().item()
        train_preds = torch.norm(caps_output, dim=-1).argmax(dim=1)
        train_correct += (train_preds == labels).sum().item()

    train_accuracy = train_correct / len(train_loader.dataset)
    avg_train_loss = train_loss / len(train_loader.dataset)
    os.system('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader')

    # === Validate ===
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            caps_output, reconstructions, pred_y = model(images)
            one_hot_labels = F.one_hot(labels, num_classes=config.run_config['NUM_CLASSES']).float()
            loss = criterion(caps_output, one_hot_labels, images, reconstructions)

            val_loss += loss.item() * images.size(0)
            # val_correct += (outputs.argmax(1) == labels).sum().item()
            val_preds = torch.norm(caps_output, dim=-1).argmax(dim=1)
            val_correct += (val_preds == labels).sum().item()

    val_accuracy = val_correct / len(val_loader.dataset)
    avg_val_loss = val_loss / len(val_loader.dataset)

    scheduler.step()

    # Log
    print(f"🧠 Train Loss: {avg_train_loss:.4f} | Accuracy: {train_accuracy:.4f}")
    print(f"🧪 Val   Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.4f}")

    # Write results to CSV
    with open(log_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy])

    # Save best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
        patience_counter = 0
        print("✅ Best model saved!")
    else:
        patience_counter += 1
        print(f"‼️No improvement in val_accuracy for {patience_counter}")
    # Early stopping
    if patience_counter >= patience:
        print(f"⚠️ Early stopping triggered after {epoch + 1} epochs")
        break

# train finishes
print(f"✅ Training completed! Best validation accuracy {best_val_accuracy:.4f}")
