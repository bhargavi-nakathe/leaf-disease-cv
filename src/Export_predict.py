import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import argparse
import os

# Argument parsing for multiple images
parser = argparse.ArgumentParser()
parser.add_argument("--images", nargs='+', required=True, help="List of paths to input images")
args = parser.parse_args()

# Load model structure
model = models.resnet18()
num_classes = 6  # Update for six classes
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load the trained weights (ResNet epoch 3)
#checkpoint_path = 'models/ResNet18_checkpoint_epoch3.pth'  # Update with your checkpoint path
checkpoint = torch.load(
    'models/ResNet18_checkpoint_epoch3.pth',
    map_location='cpu'
)

model.load_state_dict(
    checkpoint['model_state']
)

model.eval()

# Define transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class mapping
class_names = {
    0: 'Pepper,_bell___Bacterial_spot',
    1: 'Pepper,_bell___healthy',
    2: 'Tomato___Bacterial_spot',
    3: 'Tomato___Early_blight',
    4: 'Tomato___healthy',
    5: 'Tomato___Late_blight'
}

# Loop over each image path and predict
for image_path in args.images:
    img = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    print(f"Image: {image_path}")
    print(f"Prediction: {class_names[predicted_class.item()]}")
    print(f"Confidence: {confidence.item() * 100:.2f}%\n")