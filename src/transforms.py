import torchvision.transforms as transforms

# Training transforms: apply random augmentations to simulate variety
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to a consistent size
    transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
    transforms.RandomRotation(20),  # Randomly rotate by ±20 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color adjustments
    transforms.ToTensor(),  # Convert image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet means/stds
])

# Validation transforms: only deterministic steps
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to same size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Same normalization
])