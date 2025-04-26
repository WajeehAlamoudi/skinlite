import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
from sklearn.metrics import f1_score
from torch import nn
from tqdm import tqdm
import setting
from capsule.caps_layer import *
from capsule.Hcaps_data_loader import *
from mobile.mobile_data_loader import *
from mobile.mobile_model import *
from transforms import *
from torch.utils.data import DataLoader
from utils import *
from datetime import datetime
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if setting.model == "mobile":
    print(f"♦️♦️start training a {setting.model} model♦️♦️")
    # 1. Load Data
    train_set = ISICDataset(
        csv_path=setting.TRAIN_LABELS_DIR,
        img_dir=setting.TRAIN_IMG_DIR,
        set_state='train',
        transform=train_transform
    )
    train_loader = DataLoader(train_set, batch_size=setting.BATCH_SIZE, shuffle=setting.TRAIN_SHUFFLE,
                              num_workers=setting.NUM_WORKERS)
    val_set = ISICDataset(
        csv_path=setting.VAL_LABELS_DIR,
        img_dir=setting.VAL_IMG_DIR,
        transform=val_transform,
        set_state='val'
    )
    val_loader = DataLoader(val_set, batch_size=setting.BATCH_SIZE, shuffle=False, num_workers=setting.NUM_WORKERS)

    class_weights = compute_loader_class_weights(train_set).to(setting.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = MobileNetClassifier(num_classes=setting.CLASSES_LEN).to(setting.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=setting.LEARNING_RATE)

    EPOCHS = setting.EPOCHS
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    early_stop_counter = 0
    best_f1 = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        preds_all, labels_all = [], []

        for x, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x, labels = x.to(setting.DEVICE), labels.to(setting.DEVICE)

            outputs = model(x)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

        train_acc = correct / total
        avg_train_loss = running_loss / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for x, labels in val_loader:
                x, labels = x.to(setting.DEVICE), labels.to(setting.DEVICE)
                outputs = model(x)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * x.size(0)

                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total
        val_f1 = f1_score(val_labels, val_preds, average="macro")

        if val_f1 > best_f1:
            best_f1 = val_f1
            early_stop_counter = 0
            torch.save(model.state_dict(), f"best_{setting.model}_{timestamp}.pth")
            torch.save(model, f"entire_model_{setting.model}_{timestamp}.pt")
            print("✅ Model saved with improved F1:", round(best_f1, 4))
        else:
            early_stop_counter += 1
            print(f"♦️ No improvement in F1. Patience counter: {early_stop_counter}/{setting.PATIENCE}")
            if early_stop_counter >= setting.PATIENCE:
                print("⛔️ Early stopping triggered.")
                history_filename = f"{setting.model.lower()}_{timestamp}_history.pkl"
                with open(history_filename, 'wb') as f:
                    pickle.dump(history, f)
                print(f"♦️♦️history of {setting.model} model training saved by name {history_filename} ♦️♦️")
                break

        # Save metrics
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss / val_total)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}")
        print(f"Val Loss: {val_loss / val_total:.4f} | Val Acc: {val_acc:.2f} | Val F1: {val_f1:.4f}")

        # Automatically choose name based on model type
        history_filename = f"{setting.model.lower()}_{timestamp}_history.pkl"
        with open(history_filename, 'wb') as f:
            pickle.dump(history, f)
    print(f"♦️♦️ history of training {setting.model} model saved♦️♦️")

if setting.model == "Hcaps":
    print(f"♦️♦️start training a {setting.model} model♦️♦️")
    # 1. Load Data
    train_set = HCAPS_ISICDataset(
        csv_path=setting.TRAIN_LABELS_DIR,
        img_dir=setting.TRAIN_IMG_DIR,
        set_state='train',
        transform=train_transform
    )
    train_loader = DataLoader(train_set, batch_size=setting.BATCH_SIZE, shuffle=setting.TRAIN_SHUFFLE,
                              num_workers=setting.NUM_WORKERS)
    val_set = HCAPS_ISICDataset(
        csv_path=setting.VAL_LABELS_DIR,
        img_dir=setting.VAL_IMG_DIR,
        transform=val_transform,
        set_state='val'
    )
    val_loader = DataLoader(val_set, batch_size=setting.BATCH_SIZE, shuffle=False, num_workers=setting.NUM_WORKERS)

    # 2. Initialize Model, Loss, Optimizer
    model = HCapsNet().to(setting.DEVICE)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=setting.LEARNING_RATE)
    EPOCHS = setting.EPOCHS
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": [], "gamma": []}
    early_stop_counter = 0
    best_f1 = 0
    # 3. Training Loop
    for epoch in range(1, EPOCHS + 1):

        # === Update learning rate with exponential decay ===
        if epoch > setting.KAPPA:
            new_lr = setting.LEARNING_RATE * (setting.BETA ** (epoch - setting.KAPPA))
        else:
            new_lr = setting.LEARNING_RATE
        # Apply the new learning rate to the optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"♦️ Epoch {epoch} - Learning Rate: {new_lr:.6f}")

        model.train()
        running_loss, corrects, total = 0, [0, 0, 0], 0
        preds_all, labels_all = [[], [], []], [[], [], []]

        for x, label1, label2, label3 in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x = x.to(setting.DEVICE)
            label1, label2, label3 = label1.to(setting.DEVICE), label2.to(setting.DEVICE), label3.to(setting.DEVICE)
            labels = (label1, label2, label3)

            outputs = model(x)
            gamma = [1 / 3, 1 / 3, 1 / 3]  # initial default
            loss, _, _, _ = total_hcapsnet_loss(outputs, labels, x, gamma, setting.LAMBDA_RECON, class_weights=None)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            for i, (d, l) in enumerate(zip([outputs["digit1"], outputs["digit2"], outputs["digit3"]], labels)):
                pred = d.norm(dim=-1).argmax(dim=1)
                corrects[i] += (pred == l).sum().item()
                preds_all[i].extend(pred.cpu().numpy())
                labels_all[i].extend(l.cpu().numpy())
            total += x.size(0)

        accs = [c / total for c in corrects]
        gammas = dynamic_gamma(setting.H_CLASSES_COUNT, accs, setting.LAMBDA_RECON)

        # --- Validation ---
        model.eval()
        val_corrects, val_total = [0, 0, 0], 0
        val_preds, val_labels = [[], [], []], [[], [], []]
        val_loss_total = 0

        with torch.no_grad():
            for x, label1, label2, label3 in val_loader:
                x = x.to(setting.DEVICE)
                label1, label2, label3 = label1.to(setting.DEVICE), label2.to(setting.DEVICE), label3.to(setting.DEVICE)
                labels = (label1, label2, label3)

                outputs = model(x)
                loss, _, _, _ = total_hcapsnet_loss(outputs, labels, x, gammas, setting.LAMBDA_RECON)
                val_loss_total += loss.item() * x.size(0)

                for i, (d, l) in enumerate(zip([outputs["digit1"], outputs["digit2"], outputs["digit3"]], labels)):
                    pred = d.norm(dim=-1).argmax(dim=1)
                    val_corrects[i] += (pred == l).sum().item()
                    val_preds[i].extend(pred.cpu().numpy())
                    val_labels[i].extend(l.cpu().numpy())
                val_total += x.size(0)

        avg_train_loss = running_loss / total
        avg_val_loss = val_loss_total / val_total
        val_accs = [c / val_total for c in val_corrects]
        val_f1_macro = f1_score(val_labels[2], val_preds[2], average='macro')

        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            early_stop_counter = 0
            torch.save(model.state_dict(), f"best_{setting.model}_{timestamp}.pth")
            torch.save(model, f"entire_model_{setting.model}_{timestamp}.pt")
            print("✅ Model saved with improved F1:", round(best_f1, 4))
        else:
            early_stop_counter += 1
            print(f"♦️ No improvement in F1. Patience counter: {early_stop_counter}/{setting.PATIENCE}")
            if early_stop_counter >= setting.PATIENCE:
                print("⛔️ Early stopping triggered.")
                history_filename = f"{setting.model.lower()}_{timestamp}_history.pkl"
                with open(history_filename, 'wb') as f:
                    pickle.dump(history, f)
                print(f"♦️♦️history of {setting.model} model training saved by name {history_filename} ♦️♦️")
                break

        # --- Logging ---
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(accs)  # ✅ store Acc1, Acc2, Acc3
        history["val_acc"].append(val_accs)
        history["val_f1"].append(val_f1_macro)
        history["gamma"].append(gammas)

        print(f"Train Loss: {avg_train_loss:.4f} | Acc1: {accs[0]:.2f}, Acc2: {accs[1]:.2f}, Acc3: {accs[2]:.2f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accs[2]:.2f} | Val F1 (fine): {val_f1_macro:.4f}")
        print(f"γ weights: {gammas}")

        # Automatically choose name based on model type
        history_filename = f"{setting.model.lower()}_{timestamp}_history.pkl"
        with open(history_filename, 'wb') as f:
            pickle.dump(history, f)

    print(f"♦️♦️ history of training {setting.model} model saved♦️♦️")

if setting.model not in ["Hcaps", "mobile"]:
    print(f"The model {setting.model} is not supported, Check the setting file.")
