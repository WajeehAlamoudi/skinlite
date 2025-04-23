import torch

import config
from models.model import build_model

model = build_model(
    arch=config.run_config['MODEL_ARCH'],
    trainable_layers=config.run_config['TRAINABLE_LAYERS'],
    pretrained=config.run_config['PRE_TRAINED'],
    num_primary_units=8,
    primary_unit_size=32,
    num_classes=7
)

# Dummy input: [batch_size, channels, height, width]
dummy_input = torch.randn(1, 3, 224, 224)

# Pass through model
model(dummy_input)