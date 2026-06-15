import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
from transforms import train_transforms 
from torchvision.datasets import ImageFolder # Import your transforms



# Load the original image (single leaf)
dataset = ImageFolder("data/train")

image, label = dataset[0]

# Apply transforms 8 times and collect them
augmented_images = []
for _ in range(8):
    augmented_image = train_transforms(image)  # Apply the train transform
    augmented_images.append(augmented_image)

# Make a grid from the 8 images
grid = make_grid(augmented_images, nrow=4, normalize=True)  # 4x2 grid

# Save the grid
plt.imshow(grid.permute(1, 2, 0).detach().numpy())  # Convert to HWC for plotting
plt.axis('off')
plt.savefig('report/augmented_leaf_grid.png')  # Save to file
plt.show()