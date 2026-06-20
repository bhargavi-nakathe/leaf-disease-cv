import sys
import os
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('src'))
from dataset import val_loader

# ----------------------------
# Setup
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("reports", exist_ok=True)

# ----------------------------
# Load model
# ----------------------------
model = models.resnet18(pretrained=False)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 6)

checkpoint = torch.load(
    "models/ResNet18_checkpoint_epoch3.pth",
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model_state"])

model = model.to(DEVICE)
model.eval()

# ----------------------------
# Prediction collection
# ----------------------------
all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# ----------------------------
# Classification report
# ----------------------------
class_names = val_loader.dataset.classes

report = classification_report(
    all_labels,
    all_preds,
    target_names=class_names
)

print(report)

with open("reports/classification_report.txt", "w") as f:
    f.write(report)

print("Classification report saved.")

# ----------------------------
# Confusion matrix
# ----------------------------
cm = confusion_matrix(
    all_labels,
    all_preds
)

plt.figure(figsize=(8, 6))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.colorbar()

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    range(len(class_names)),
    class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "reports/confusion_matrix.png",
    dpi=150
)

print("Confusion matrix saved.")