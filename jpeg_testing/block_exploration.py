import hashlib
from io import BytesIO
import numpy as np
import torch
import argparse
from torchvision import datasets, transforms
from PIL import Image
import torchvision

from sklearn.manifold import TSNE

def get_box(image, box_size):
    return torch.floor(image / box_size).flatten()

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
    image = torchvision.transforms.functional.to_pil_image(tensor[0])

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

def get_hash(tensor):
    sha256 = hashlib.sha256()
    tensor_bytes = tensor.to(torch.int8).numpy().tobytes()
    sha256.update(tensor_bytes)
    return sha256.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("num_blocks", type=int)

    args = parser.parse_args()

    block_size = 1 / args.num_blocks

    train_images = datasets.ImageFolder(args.data_path, transforms.Compose([
                transforms.Resize(224),
                transforms.RandomCrop((224, 224), pad_if_needed=True),
                transforms.ToTensor(),
            ]))
    
    loader = torch.utils.data.DataLoader(train_images, batch_size=1, shuffle=True,
                                               num_workers=2)

    print("Dataloader Created")

    vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
    vgg16.eval()

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for image, _ in loader:
        compressed = jpeg_compress_tensor(image, 90)

        embedding1 = torch.nn.functional.sigmoid(vgg16(norm(image)))
        embedding2 = torch.nn.functional.sigmoid(vgg16(norm(compressed.unsqueeze(0))))

        print(image - compressed)
        print(embedding1 - embedding2)

        print(embedding1.max())
        print(embedding1.min())

        print((embedding1 - embedding2).max())
        print(((embedding1 - embedding2)**2).sum())

        break

        both = torch.stack([image[0], compressed])
        torchvision.utils.save_image(both, "temp.jpg")

        image = image[0]
        box_1 = get_box(image, block_size)
        box_2 = get_box(compressed, block_size)
        
        print(box_1, box_2)

        # print((box_2-box_1).sum())

        hash_1 = get_hash(box_1)
        hash_2 = get_hash(box_2)

        print(hash_1)
        print(hash_2)

        print(hash_1 == hash_2)
    
    # print(f"Conflicts: {len(vals) - len(val_set)}")

main()