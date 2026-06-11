import os
import shutil
import random
from collections import defaultdict

# paths
RAW_DIR   = "data/raw/"
TRAIN_DIR = "data/train/"
VAL_DIR   = "data/val/"

TRAIN_RATIO = 0.8
SEED        = 42

random.seed(SEED)

# step 1: loop through each class folder
for class_name in os.listdir(RAW_DIR):
    class_path = os.path.join(RAW_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    # step 2: get all image files in this class
    images = os.listdir(class_path)
    random.shuffle(images)

    # step 3: calculate split point
    split_point = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_point]
    val_images   = images[split_point:]

    # step 4: create destination folders
    train_class_dir = os.path.join(TRAIN_DIR, class_name)
    val_class_dir   = os.path.join(VAL_DIR,   class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir,   exist_ok=True)

    # step 5: copy images (not move — keeps raw/ intact)
    for img in train_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(train_class_dir, img)
        )
    for img in val_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(val_class_dir, img)
        )

    print(f"{class_name}")
    print(f"  train: {len(train_images)}  val: {len(val_images)}")

print("\nDone!")
print(f"Train folder: {TRAIN_DIR}")
print(f"Val folder:   {VAL_DIR}")