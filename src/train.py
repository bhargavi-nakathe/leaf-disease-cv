import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import SmallCNN
from dataloader import train_loader, val_loader

# ── setup ──────────────────────────────────────────────
EPOCHS     = 3
LR         = 0.001
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Training on: {DEVICE}")

model     = SmallCNN(num_classes=6).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ── tracking ───────────────────────────────────────────
train_losses = []
val_losses   = []
val_accuracies = []

# ── training loop ──────────────────────────────────────
for epoch in range(EPOCHS):

    # ── train phase ────────────────────────────────────
    model.train()       # tells model: training mode ON
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        predictions = model(images)           # forward pass
        loss        = criterion(predictions, labels)  # calculate loss

        optimizer.zero_grad()   # clear old gradients
        loss.backward()         # backward pass
        optimizer.step()        # update weights

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ── val phase ──────────────────────────────────────
    model.eval()        # tells model: training mode OFF
    val_loss    = 0.0
    correct     = 0
    total       = 0

    with torch.no_grad():   # no gradients needed for val
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            predictions = model(images)
            loss        = criterion(predictions, labels)
            val_loss   += loss.item()

            # accuracy — which class had highest score
            predicted_classes = predictions.argmax(dim=1)
            correct += (predicted_classes == labels).sum().item()
            total   += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    accuracy     = correct / total * 100

    val_losses.append(avg_val_loss)
    val_accuracies.append(accuracy)

    print(f"Epoch {epoch+1}/{EPOCHS} "
          f"| train loss: {avg_train_loss:.4f} "
          f"| val loss: {avg_val_loss:.4f} "
          f"| val accuracy: {accuracy:.2f}%")

    # ── save checkpoint every epoch ────────────────────
    checkpoint = {
        "epoch"      : epoch + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_accuracy"   : accuracy,
    }
    torch.save(checkpoint, f"{MODEL_DIR}/checkpoint_epoch{epoch+1}.pth")
    print(f"  checkpoint saved → models/checkpoint_epoch{epoch+1}.pth")

# ── plot loss curve ────────────────────────────────────
epochs_range = range(1, EPOCHS + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs_range, train_losses, label="Train loss")
ax1.plot(epochs_range, val_losses,   label="Val loss")
ax1.set_title("Loss curve")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(epochs_range, val_accuracies, label="Val accuracy", color="green")
ax2.set_title("Validation accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy %")
ax2.legend()

plt.tight_layout()
plt.savefig("track/loss_curve.png", dpi=150)
print("\nPlot saved → track/loss_curve.png")