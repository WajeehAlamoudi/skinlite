import os
import torch
import yaml
from torch.utils.data import DataLoader

from data.isic_loader import ISICDataset
from models.model import build_model


def load_test_config(config_path):
    """
    Load only valid test-related entries from config.yaml, skipping unserializable ones.
    """
    with open(config_path, 'r') as f:
        raw_text = f.read()

    cleaned_lines = []
    skip_block = False
    for line in raw_text.splitlines():
        if '!!python/object/apply' in line:
            skip_block = True
            continue
        elif skip_block and not line.startswith(' '):
            skip_block = False

        if not skip_block:
            cleaned_lines.append(line)

    clean_yaml = "\n".join(cleaned_lines)
    cfg = yaml.safe_load(clean_yaml)

    return {
        'IMAGE_SIZE': cfg['IMAGE_SIZE'],
        'NUM_CLASSES': cfg['NUM_CLASSES'],
        'BATCH_SIZE': cfg['BATCH_SIZE'],
        'NUM_WORKERS': cfg.get('NUM_WORKERS', 4),
        'MODEL_ARCH': cfg['MODEL_ARCH'],
        'TRAINABLE_LAYERS': 0  # always frozen during test
    }


def test_model(head_dir):
    config_path = os.path.join(head_dir, 'config.yaml')
    checkpoint_path = os.path.join(head_dir, 'best_model.pth')

    if not os.path.isfile(config_path) or not os.path.isfile(checkpoint_path):
        print("❌ Missing config.yaml or best_model.pth in:", head_dir)
        return

    config = load_test_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    test_dataset = ISICDataset(set_state='test', output_size=config['IMAGE_SIZE'])
    test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], num_workers=config['NUM_WORKERS'])

    # Model
    model = build_model(
        arch=config['MODEL_ARCH'],
        input_size=config['IMAGE_SIZE'],
        num_classes=config['NUM_CLASSES'],
        trainable_layers=config['TRAINABLE_LAYERS'],  # 0 during testing
        pretrained=False
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Evaluation
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"\n📂 Head Directory: {head_dir}")
    print(f"🧪 Test Accuracy: {correct / total:.4f}")


# Example usage
if __name__ == "__main__":
    head_dir = r"C:\Users\wajee\Downloads\run_20250419_1147"  # Replace with your path
    test_model(head_dir)
