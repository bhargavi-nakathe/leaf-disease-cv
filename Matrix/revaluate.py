import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import models
from torchvision.utils import save_image

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.abspath('src'))
from dataset import val_loader

# ----------------------------------
# Setup
# ----------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("report", exist_ok=True)
os.makedirs("report/errors", exist_ok=True)

print(f"Evaluating on: {DEVICE}")

# ----------------------------------
# Load model
# ----------------------------------
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

# ----------------------------------
# Evaluation
# ----------------------------------
all_preds = []
all_labels = []

error_count = 0
MAX_ERRORS = 5

class_names = val_loader.dataset.classes

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Save misclassified images
        for i in range(len(images)):

            true_label = labels[i].item()
            pred_label = preds[i].item()

            if true_label != pred_label and error_count < MAX_ERRORS:

                filename = (
                    f"report/errors/"
                    f"true_{class_names[true_label]}"
                    f"_pred_{class_names[pred_label]}"
                    f"_{error_count+1}.png"
                )

                save_image(images[i].cpu(), filename)

                print(f"Saved: {filename}")

                error_count += 1

# ----------------------------------
# Classification Report
# ----------------------------------
report = classification_report(
    all_labels,
    all_preds,
    target_names=class_names
)

print("\nClassification Report:\n")
print(report)

with open(
    "report/classification_report2.txt",
    "w"
) as f:
    f.write(report)

print("Classification report saved.")

# ----------------------------------
# Confusion Matrix
# ----------------------------------
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
    "report/confusion_matrix2.png",
    dpi=150
)

print("Confusion matrix saved.")
print(f"Saved {error_count} misclassified examples.")