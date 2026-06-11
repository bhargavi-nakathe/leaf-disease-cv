import torch
import torchvision
import PIL
import matplotlib


def main():
    print("PyTorch Environment Verification")
    print("--------------------------------")

    print(f"torch version       : {torch.__version__}")
    print(f"torchvision version : {torchvision.__version__}")
    print(f"Pillow version      : {PIL.__version__}")
    print(f"matplotlib version  : {matplotlib.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available      : {cuda_available}")

    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        device = torch.device("cuda")
    else:
        device_name = "CPU"
        device = torch.device("cpu")

    print(f"Device name         : {device_name}")

    sample_tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
    print(f"Tensor device       : {sample_tensor.device}")

    print("--------------------------------")
    print("Environment verification completed successfully.")


if __name__ == "__main__":
    main()