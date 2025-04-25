from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
import again.setting

CLASS_NAMES = again.setting.CLASS_NAMES
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
CLASS_MULTIPLIER = again.setting.CLASS_MULTIPLIER


class HCAPS_ISICDataset(Dataset):
    def __init__(self, csv_path, img_dir, set_state, transform=None):
        self.transform = transform
        self.set_state = set_state.lower()
        self.img_dir = img_dir

        df = pd.read_csv(csv_path)
        base_samples = []
        for _, row in df.iterrows():
            image_id = row['image']
            label_name = row[CLASS_NAMES].astype(int).idxmax()
            label = LABEL_MAP[label_name]
            base_samples.append((f"{image_id}.jpg", label))

        # === Expand or reduce samples for training
        if self.set_state == 'train':
            class_img_dict = {}
            for path, label in base_samples:
                class_img_dict.setdefault(label, []).append(path)

            expanded = []
            for label, paths in class_img_dict.items():
                multiplier = CLASS_MULTIPLIER.get(label, 1)

                if multiplier > 0:
                    expanded.extend([(path, label) for path in paths for _ in range(multiplier)])
                elif multiplier < 0:
                    keep_every = abs(multiplier)
                    reduced = paths[::keep_every]
                    expanded.extend([(path, label) for path in reduced])
                else:
                    expanded.extend([(path, label) for path in paths])

            self.samples = expanded

        elif self.set_state in ['val', 'test']:
            self.samples = base_samples
        else:
            raise ValueError("Invalid set_state. Choose from ['train', 'val', 'test']")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, fine_label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Define hierarchy mapping
        coarse_map = {
            0: 0,  # MEL → Malignant
            1: 1,  # NV  → Benign
            2: 0,  # BCC → Malignant
            3: 0,  # AKIEC → Malignant
            4: 1,  # BKL → Benign
            5: 1,  # DF  → Benign
            6: 1   # VASC → Benign
        }
        medium_map = {
            0: 0,  # MEL
            1: 2,  # NV  → Other
            2: 1,  # BCC
            3: 1,  # AKIEC
            4: 2,  # BKL → Other
            5: 2,  # DF  → Other
            6: 2  # VASC → Other
        }

        label1 = coarse_map[fine_label]  # 0 or 1
        label2 = medium_map[fine_label]  # 0, 1, or 2
        label3 = fine_label  # 0–6

        return image, label1, label2, label3
