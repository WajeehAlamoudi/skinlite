from collections import Counter
import torch
import setting


def compute_loader_class_weights(dataset, num_classes=setting.CLASSES_LEN):
    label_counts = Counter(label for _, label in dataset.samples)
    total = sum(label_counts.values())
    weights = [total / (num_classes * label_counts.get(i, 1)) for i in range(num_classes)]
    normed = torch.tensor(weights, dtype=torch.float32)
    return normed / normed.sum()



