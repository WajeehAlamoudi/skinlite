import matplotlib.pyplot as plt
import pickle
import os
import argparse

# === Argument parsing ===
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, choices=['Hcaps', 'mobile'], help="Model name: Hcaps or mobile")
parser.add_argument('--history_path', type=str, default=None, help="Path to history .pkl file")
args = parser.parse_args()

# === Detect and load history ===
default_paths = {
    "Hcaps": "hcaps_history.pkl",
    "mobile": "mobile_history.pkl"
}

history_file = args.history_path or default_paths[args.model]

if not os.path.exists(history_file):
    raise FileNotFoundError(f"History file not found: {history_file}")

with open(history_file, "rb") as f:
    history = pickle.load(f)

# === Plot shared metrics ===
plt.figure(figsize=(10, 5))
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.title(f"{args.model} - Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot model-specific metrics ===
if args.model == "Hcaps":
    # Val Acc per hierarchy
    accs = list(zip(*history["val_acc"]))  # [[acc1s], [acc2s], [acc3s]]
    plt.figure(figsize=(10, 5))
    plt.plot(accs[0], label="Val Acc - Coarse")
    plt.plot(accs[1], label="Val Acc - Medium")
    plt.plot(accs[2], label="Val Acc - Fine")
    plt.title("Validation Accuracy per Hierarchy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # F1
    plt.figure(figsize=(10, 5))
    plt.plot(history["val_f1"], label="Val F1 - Fine")
    plt.title("Fine-Level Macro F1 per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Gamma weights
    gamma = list(zip(*history["gamma"]))  # γ1, γ2, γ3
    plt.figure(figsize=(10, 5))
    plt.plot(gamma[0], label="γ1 (Coarse)")
    plt.plot(gamma[1], label="γ2 (Medium)")
    plt.plot(gamma[2], label="γ3 (Fine)")
    plt.title("Dynamic Gamma Weights per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Gamma Weight")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

elif args.model == "mobile":
    # Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Val Accuracy")
    plt.title("MobileNet Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # F1 Score (if exists)
    if "val_f1" in history:
        plt.figure(figsize=(10, 5))
        plt.plot(history["val_f1"], label="Val F1 Score")
        plt.title("MobileNet Macro F1 per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
