from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter
import config
from utils.helpers import load_labeled_paths, seed_worker, compute_class_weights_from_labels
from utils.transforms import custom_transform, simple_transform

# {'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}
# original_counts = {
#     0: 1113,  # MEL
#     1: 6705,  # NV
#     2: 514,   # BCC
#     3: 327,   # AKIEC
#     4: 1099,  # BKL
#     5: 115,   # DF
#     6: 142    # VASC
# }
CLASS_MULTIPLIER = {
    6: 8,  # VASC
    5: 8,  # DF
    4: 8,  # BKL
    3: 4,  # AKIEC
    2: 2,  # BCC
    1: -2,  # NV
    0: 2,  # MEL
}


# create a custom pytorch data loader
class ISICDataset(Dataset):
    def __init__(self, set_state, output_size):
        self.set_state = set_state.lower()
        self.output_size = output_size

        if set_state == 'train':
            self.image_paths, self.label_paths = load_labeled_paths(config.TRAIN_IMG_DIR, config.TRAIN_LABELS_DIR)
            self.transform = custom_transform(output_size=self.output_size)
            # Expanded lists
            expanded_imgs, expanded_labels = [], []
            class_img_dict = {}

            # Group by label first
            for img_path, label in zip(self.image_paths, self.label_paths):
                class_img_dict.setdefault(label, []).append(img_path)

            # Expand or downsample
            for label, paths in class_img_dict.items():
                multiplier = CLASS_MULTIPLIER.get(label, 1)

                if multiplier > 0:
                    for path in paths:
                        for _ in range(multiplier):
                            expanded_imgs.append(path)
                            expanded_labels.append(label)

                elif multiplier < 0:
                    keep_every = abs(multiplier)
                    reduced = paths[::keep_every]  # keep 1 of every `abs(multiplier)`
                    expanded_imgs.extend(reduced)
                    expanded_labels.extend([label] * len(reduced))

                elif multiplier == 0:
                    # Keep the original images without duplication
                    expanded_imgs.extend(paths)
                    expanded_labels.extend([label] * len(paths))

            self.image_paths = expanded_imgs
            self.label_paths = expanded_labels
            # print("[DEBUG] 111 Training transform applied:", self.transform)

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
        image = Image.open(img_path).convert("RGB")

        # if self.transform is not None:
        # print(f"[DEBUG] 222 Transform applied on index {idx}")
        # else:
        # print(f"[WARNING] 333 No transform applied on index {idx}")

        transformed_image = self.transform(image)
        return transformed_image, label

    def get_loader(self, batch_size, num_workers):
        if self.set_state == 'train':
            # class_counts = Counter(self.label_paths)
            # class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}

            # class_weights = compute_soft_class_weights(
            #     labels=self.label_paths,
            #     num_classes=config.run_config['NUM_CLASSES'],
            #     smoothing=0.01  # 0 -> 1/frq, 0.999 -> weak balance
            # )
            # sample_weights = [class_weights[label] for label in self.label_paths]
            # sampler = WeightedRandomSampler(
            #     weights=sample_weights,
            #     num_samples=len(sample_weights),
            #     replacement=True
            # )

            return DataLoader(
                self,
                batch_size=batch_size,
                # sampler=sampler,
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker
            )

        else:
            return DataLoader(
                self,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )


"""
# raw class weight = class_weights = np.array(np.mean(Y_orig, axis = 0)).astype('float32')
Class | Raw Count | After Sampler (Approx)
NV | 6705 | 1428
MEL | 1113 | 1432
BKL | 1099 | 1430
BCC | 514 | 1427
AKIEC | 327 | 1431
DF | 115 | 1433
VASC | 142 | 1434
"""
