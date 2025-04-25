from collections import Counter
import torch
import setting
import matplotlib.pyplot as plt


def compute_loader_class_weights(dataset, num_classes=setting.CLASSES_LEN):
    label_counts = Counter(label for _, label in dataset.samples)
    total = sum(label_counts.values())
    weights = [total / (num_classes * label_counts.get(i, 1)) for i in range(num_classes)]
    if setting.debug:
        print(weights)
    normed = torch.tensor(weights, dtype=torch.float32)
    if setting.debug:
        print(normed)
    return normed / normed.sum()


def draw_train_metrics(history, metrics):
    values = [accs[2] for accs in history["val_acc"]]

    plt.figure(figsize=(8, 5))
    plt.plot(values, label="Fine-Level Val Accuracy (Acc3)", marker='o')
    plt.title("H-CapsNet Fine-Level Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()
