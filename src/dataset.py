import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class LeafDiseaseDataset(Dataset):
    """
    Reads images from folders like this:
    data/raw/
        Tomato___healthy/        <- folder name = class name
            image1.jpg
            image2.jpg
        Tomato___Early_blight/
            image1.jpg
        Pepper,_bell___healthy/
            image1.jpg
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir      # path to data/raw/
        self.transform = transform    # image transformations
        self.samples = []             # will hold (image_path, label) pairs
        self.classes = []             # will hold class names

        # Step 1: get all folder names (each folder = one class)
        self.classes = sorted(os.listdir(root_dir))

        # Step 2: class name → number  (e.g. "Tomato___healthy" → 0)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Step 3: walk through each folder and collect image paths
        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            for img_file in os.listdir(cls_folder):
                img_path = os.path.join(cls_folder, img_file)
                label = self.class_to_idx[cls]
                self.samples.append((img_path, label))
        

    def __len__(self):
        # how many total images
        return len(self.samples)

    def __getitem__(self, idx):
        # load one image and its label by index
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label