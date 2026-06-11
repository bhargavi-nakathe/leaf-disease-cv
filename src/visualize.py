import matplotlib.pyplot as plt
import torchvision
from dataloader import train_loader, train_dataset

# Step 1: grab one batch from the train loader
imgs, labels = next(iter(train_loader))

print("Batch shape:", imgs.shape)  # (32, 3, 224, 224)

# Step 2: show 16 images in a 4x4 grid
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle(f"Sample Batch — shape: {tuple(imgs.shape)}", fontsize=13)

for i, ax in enumerate(axes.flatten()):
    img = imgs[i]

    # undo normalisation so colours look correct
    img = img * 0.225 + 0.45

    # convert from (3, H, W) → (H, W, 3) for matplotlib
    img = img.permute(1, 2, 0).clamp(0, 1)

    class_name = train_dataset.classes[labels[i].item()]
    short_name = class_name.replace("Tomato___", "").replace("Pepper,_bell___", "Pepper ")

    ax.imshow(img)
    ax.set_title(short_name, fontsize=7)
    ax.axis("off")

plt.tight_layout()
plt.savefig("track/task2.png", dpi=150)
print("Saved to track/task2.png")
plt.show()