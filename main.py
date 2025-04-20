import torch

import config
from data.isic_loader import ISICDataset
from utils.helpers import compute_soft_class_weights
from utils.loss_functions import CustomLoss

if __name__ == '__main__':
    train_dataset = ISICDataset(
        set_state='train',
        output_size=config.run_config['IMAGE_SIZE']
    )

    train_loader = train_dataset.get_loader(
        batch_size=config.run_config['BATCH_SIZE'],
        num_workers=config.run_config['NUM_WORKERS']
    )

    for i, (images, labels) in enumerate(train_loader):
        print(f"\n🟦 Batch {i+1} Labels:")
        print(labels)
        if i == 15:
            break


