from io import BytesIO
import os
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
import argparse

def jpeg_compress_tensor(tensor: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """
    Compress a PyTorch tensor using JPEG and return the result as a new tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor with shape (C, H, W) in the range [0, 1].
        quality (int): JPEG compression quality (1-100).
    
    Returns:
        torch.Tensor: JPEG-compressed tensor.
    """
    # Ensure tensor is on CPU and in the range [0, 255]
    image = torchvision.transforms.functional.to_pil_image(tensor)

    # Save to a BytesIO buffer with JPEG compression
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    compressed_image = Image.open(buffer)
    
    # Convert back to tensor
    compressed_tensor = torch.from_numpy(np.array(compressed_image))
    
    # Convert back to channel-first format (C, H, W)
    if compressed_tensor.dim() == 3:  # Assuming channel-last
        compressed_tensor = compressed_tensor.permute(2, 0, 1)
    
    # Normalize back to [0, 1]
    compressed_tensor = compressed_tensor.float() / 255.0
    
    return compressed_tensor

def jpeg_compress_tensor_multiple(tensors: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """
    Compress a list of PyTorch tensors using JPEG and return the result as a new tensor.
    
    Args:
        tensors (torch.Tensor): Input tensor with shape (N, C, H, W) in the range [0, 1].
        quality (int): JPEG compression quality (1-100).
    
    Returns:
        torch.Tensor: JPEG-compressed tensor.
    """
    return torch.stack([jpeg_compress_tensor(t, quality) for t in tensors])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    # parser.add_argument("model_path", type=str)
    args = parser.parse_args()

    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=3, init_features=32, pretrained=False)

    train_images = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop((256, 256), pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]))
    val_images = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]))

    train_loader = torch.utils.data.DataLoader(train_images, batch_size=4, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_images, batch_size=4, shuffle=False, num_workers=4)

    device = "cuda"
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(50):
        tot_mse = 0
        for i, (image, __) in enumerate(train_loader):
            image = image.to(device)
            output = model(image)

            jpeg_image = jpeg_compress_tensor_multiple(image * 2 - 1, quality = 50).to(device)

            mse = torch.nn.functional.mse_loss(output, jpeg_image)
            tot_mse += mse.item()

            optimizer.zero_grad()
            mse.backward()
            optimizer.step()
        
        print(f"[Train] Epoch {epoch}, MSE: {tot_mse / len(train_loader) / 4}")

        tot_mse = 0
        for i, (image, __) in enumerate(val_loader):
            image = image.to(device)
            output = model(image)

            jpeg_image = jpeg_compress_tensor_multiple(image * 2 - 1, quality = 50).to(device)

            mse = torch.nn.functional.mse_loss(output, jpeg_image)
            tot_mse += mse.item()

        torch.save(model.state_dict(), f"jpeg/jpeg_{epoch}.pth")

        



if __name__ == "__main__":
    main()