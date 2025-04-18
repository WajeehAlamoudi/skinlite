import csv

import torch
from torch import nn

import config
from data.isic_loader import ISICDataset
import os
from utils.helpers import setup_run_folder
from models.model import build_model
from models.optimizer import get_optimizer

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
    input_size=config.run_config['IMAGE_SIZE'],
    num_classes=config.run_config['NUM_CLASSES'],
    trainable_layers=config.run_config['TRAINABLE_LAYERS'],
    pretrained=config.run_config['PRE_TRAINED']
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
criterion = nn.CrossEntropyLoss()

num_epochs = config.run_config['EPOCH']
best_val_accuracy = 0.0

# 🔸 Step 6: Train
for epoch in range(num_epochs):
    print(f"\n🔁 Epoch {epoch+1}/{num_epochs}")

    # === Train ===
    model.train()
    train_loss, train_correct = 0.0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()

    train_accuracy = train_correct / len(train_loader.dataset)
    avg_train_loss = train_loss / len(train_loader.dataset)

    # === Validate ===
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()

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
        print("✅ Best model saved!")

# Ready to train
print("✅ Training setup complete. Ready to begin training.")
