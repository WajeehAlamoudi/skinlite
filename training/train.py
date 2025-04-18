import torch
from torch import nn

import config
from data.isic_loader import ISICDataset
import os
from datetime import datetime
from torchvision.transforms.functional import to_pil_image
from models.model import build_model
from models.optimizer import get_optimizer

# 🔸 Step 1: Create a run folder
run_name = datetime.now().strftime("run_%Y%m%d_%H%M")
run_dir = os.path.join(config.BASE_DIR, "runs", run_name)
os.makedirs(run_dir, exist_ok=True)
print(f"📁 Created run directory at: {run_dir}")

# 🔸 Step 2: Load dataset
train_dataset = ISICDataset(set_state='train', output_size=config.run_config['IMAGE_SIZE'])
val_dataset = ISICDataset(set_state='val', output_size=config.run_config['IMAGE_SIZE'])

# 🔸 Step 2.1: show sample
original_image, transformed_image, label = train_dataset.__getitem__(7)
print(f'val dataset samples: {train_dataset.__len__()}')
# Save both
original_save_path = os.path.join(run_dir, f"original_label{label}.jpg")
transformed_save_path = os.path.join(run_dir, f"transformed_label{label}.jpg")
original_image.save(original_save_path)  # This is already a PIL.Image
to_pil_image(transformed_image).save(transformed_save_path)
print(f"✅ Saved original to:     {original_save_path}")
print(f"✅ Saved transformed to: {transformed_save_path}")


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
    lr_step=config.run_config['LR_STEP']
)

# criterion = CrossEntropyLoss()

# 🔸 Step 5: Setup device and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
# 🔸 Step 6: Optionally save config to run_dir
# import yaml
# with open(os.path.join(run_dir, "config.yaml"), "w") as f:
#     yaml.dump(config.run_config, f)

# 🔸 Step 7: Setup logging (e.g., TensorBoard)
# writer = SummaryWriter(log_dir=run_dir)

# Ready to train
print("✅ Training setup complete. Ready to begin training.")
