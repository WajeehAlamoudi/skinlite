print(f"♦️♦️start training a model♦️♦️")
    # 1. Load Data
train_set = HCAPS_ISICDataset(
        csv_path=TRAIN_LABELS_DIR,
        img_dir=TRAIN_IMG_DIR,
        set_state='train',
        transform=train_transform
    )
train_loader = DataLoader(train_set, batch_size=32, shuffle=True,
                              num_workers=8)
val_set = HCAPS_ISICDataset(
        csv_path=VAL_LABELS_DIR,
        img_dir=VAL_IMG_DIR,
        transform=val_transform,
        set_state='val'
    )
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=8)

    # 2. Initialize Model, Loss, Optimizer
model = MHCapsNet().to(DEVICE)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
EPOCHS = 10
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": [], "gamma": []}
early_stop_counter = 0
best_f1 = 0
    # 3. Training Loop
for epoch in range(1, EPOCHS + 1):

        # === Update learning rate with exponential decay ===
        if epoch > 10:
            new_lr = 0.001 * (0.95 ** (epoch - 10))
        else:
            new_lr = 0.001
        # Apply the new learning rate to the optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"🔄️ Epoch {epoch} - Learning Rate: {new_lr:.6f}")

        model.train()
        running_loss, corrects, total = 0, [0, 0, 0], 0
        preds_all, labels_all = [[], [], []], [[], [], []]

        for x, label1, label2, label3 in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x = x.to(DEVICE)
            label1, label2, label3 = label1.to(DEVICE), label2.to(DEVICE), label3.to(DEVICE)
            labels = (label1, label2, label3)

            outputs = model(x)
            gamma = [1 / 3, 1 / 3, 1 / 3]  # initial default
            loss, _, _, _ = total_hcapsnet_loss(outputs, labels, x, gamma, 0.0005, class_weights=None)

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
        gammas = dynamic_gamma([2, 3, 7], accs, 0.0005)

        # --- Validation ---
        model.eval()
        val_corrects, val_total = [0, 0, 0], 0
        val_preds, val_labels = [[], [], []], [[], [], []]
        val_loss_total = 0

        with torch.no_grad():
            for x, label1, label2, label3 in val_loader:
                x = x.to(DEVICE)
                label1, label2, label3 = label1.to(DEVICE), label2.to(DEVICE), label3.to(DEVICE)
                labels = (label1, label2, label3)

                outputs = model(x)
                loss, _, _, _ = total_hcapsnet_loss(outputs, labels, x, gammas, 0.0005)
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
            torch.save(model.state_dict(), f"best_MHmodel.pth")
            torch.save(model, f"entire_model_MHmodel.pt")
            print("✅ Model saved with improved F1:", round(best_f1, 4))
        else:
            early_stop_counter += 1
            print(f"⚠️ No improvement in F1. Patience counter: {early_stop_counter}/{10}")
            if early_stop_counter >= 10:
                print("⛔️ Early stopping triggered.")
                history_filename = f"MHmodel_history.pkl"
                with open(history_filename, 'wb') as f:
                    pickle.dump(history, f)
                print(f"♦️♦️history of MHmodel model training saved by name {history_filename} ♦️♦️")
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
        history_filename = f"MHmodel_history.pkl"
        with open(history_filename, 'wb') as f:
            pickle.dump(history, f)

print(f"♦️♦️ history of training MHmodel model saved♦️♦️")
