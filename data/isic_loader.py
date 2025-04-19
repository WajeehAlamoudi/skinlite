from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter
import config
from utils.helpers import load_labeled_paths, seed_worker, compute_soft_class_weights
from utils.transforms import custom_transform, simple_transform


# create a custom pytorch data loader
class ISICDataset(Dataset):
    def __init__(self, set_state, output_size):
        self.set_state = set_state.lower()
        self.output_size = output_size
        self.shuffle = False

        if set_state == 'train':
            self.image_paths, self.label_paths = load_labeled_paths(config.TRAIN_IMG_DIR, config.TRAIN_LABELS_DIR)
            self.transform = custom_transform(output_size=self.output_size)
            #print("[DEBUG] 111 Training transform applied:", self.transform)

        elif set_state == 'val':
            self.image_paths, self.label_paths = load_labeled_paths(config.VAL_IMG_DIR, config.VAL_LABELS_DIR)
            self.transform = simple_transform(output_size=self.output_size)

        elif set_state == 'test':
            self.image_paths, self.label_paths = load_labeled_paths(config.TEST_IMG_DIR, config.TEST_LABELS_DIR)
            self.transform = simple_transform(output_size=self.output_size)

        else:
            raise ValueError("Invalid set_state. Choose from ['train', 'val', 'test']")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.label_paths[idx]
        # if self.transform is not None:
        # print(f"[DEBUG] 222 Transform applied on index {idx}")
        # else:
        # print(f"[WARNING] 333 No transform applied on index {idx}")
        image = Image.open(img_path).convert("RGB")

        transformed_image = self.transform(image)
        return transformed_image, label

    def get_loader(self, batch_size, num_workers):
        if self.set_state == 'train':
            #class_counts = Counter(self.label_paths)
            #class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
            class_weights = compute_soft_class_weights(
                labels=self.label_paths,
                num_classes=config.run_config['NUM_CLASSES'],
                smoothing=0.99  # 0 -> 1/frq, 0.999 -> weak balnce
            )
            sample_weights = [class_weights[label] for label in self.label_paths]

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False if sampler is not None else self.shuffle

            return DataLoader(
                self,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=num_workers,
                worker_init_fn=seed_worker
            )

        else:
            return DataLoader(
                self,
                batch_size=batch_size,
                shuffle=self.shuffle,
                num_workers=num_workers
            )


"""
Class | Raw Count | After Sampler (Approx)
NV | 6705 | 1428
MEL | 1113 | 1432
BKL | 1099 | 1430
BCC | 514 | 1427
AKIEC | 327 | 1431
DF | 115 | 1433
VASC | 142 | 1434
"""
