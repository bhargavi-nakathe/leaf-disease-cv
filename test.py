
import torch

ckpt = torch.load('models/ResNet18_checkpoint_epoch3.pth', map_location='cpu')

print(type(ckpt))
print(ckpt.keys())