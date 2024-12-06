from io import BytesIO
import os
import pprint
import numpy as np
from torchvision import datasets, transforms
import torchvision
import torch
import argparse
from PIL import Image
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from model.encoder_decoder import SplittyDecoderWrapper
from model.hidden import Hidden
from noise_layers.noiser import Noiser
import utils

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding

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

def batched_transform(batch, transform):
    return torch.stack([transform(img) for img in batch])


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
    parser.add_argument("model_path", type=str)
    parser.add_argument("img_size", type=int)
    args = parser.parse_args()

    norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    undo_norm = transforms.Normalize([-1, -1, -1], [2, 2, 2])


    train_images = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.RandomCrop((args.img_size, args.img_size), pad_if_needed=True),
        transforms.ToTensor(),
    ]))
    val_images = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ]))
    
    vit_transform = torchvision.transforms.Resize(224)
    vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT).cuda()
    
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=1, shuffle=True,
                                               num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_images, batch_size=1, shuffle=False, 
                                                num_workers=2)

    device = "cuda"
    this_run_folder = args.model_path
    options_file = os.path.join(this_run_folder, 'options-and-config.pickle')
    train_options, hidden_config, noise_config = utils.load_options(options_file)
    checkpoint, loaded_checkpoint_file_name = utils.load_last_checkpoint(os.path.join(this_run_folder, 'checkpoints'))
    train_options.start_epoch = checkpoint['epoch'] + 1
    train_options.train_folder = os.path.join(args.data_dir, 'train')
    train_options.validation_folder = os.path.join(args.data_dir, 'val')

    noiser = Noiser(noise_config, device)
    model = Hidden(hidden_config, device, noiser, None)
    utils.model_from_checkpoint(model, checkpoint)

    message_length = hidden_config.message_length
    message = torch.randint(0, 2, (message_length,)).float().to(device)

    decoder = SplittyDecoderWrapper(model.encoder_decoder, hidden_config)

    vit = model.vit

    pprint.pformat(vars(hidden_config))

    error = 0
    tot = 0

    MODE = "NOISE"
    QUALITY = 100
    NOISE_AMOUNT = 0.1

    # if MODE == "JPEG":
    #     print(f"Using JPEG compression with quality {QUALITY}")
    # elif MODE == "NOISE":
    #     print(f"Adding noise with amount {NOISE_AMOUNT}")

    val_a = 0
    val_b = 0
    bitwise_error = 0
    tot = 0
    for i, (images, _) in enumerate(train_loader):
        save_image(images, f"../preservation_images/{i}_original.png")
        images = images.to(device)
        normed_images = norm(images)

        messages = message.unsqueeze(0).repeat(normed_images.shape[0], 1)
        encoded_images, noised_images, decoded_messages = model.encoder_decoder(normed_images, messages)
        # encoded_images = undo_norm(encoded_images)

        # Print the mean and standard deviation of the encoded image for each channel
        # print(encoded_images.mean(dim=(0, 2, 3)), encoded_images.std(dim=(0, 2, 3)), encoded_images.min(), encoded_images.max())
        # print(images.mean(dim=(0, 2, 3)), images.std(dim=(0, 2, 3)), images.min(), images.max())

        # Simulate saving and loading the image (round it)
        # encoded_images = (encoded_images * 255).int()
        # encoded_images = encoded_images.float() / 255
        # encoded_images = torch.clamp(encoded_images, 0, 1)

        # print(encoded_images.mean(), encoded_images.min())
        # print(images.mean(), images.min(), images.max())
        # print(undo_norm(norm(images)).mean(), images.min(), images.max())

        save_image(encoded_images, f"../preservation_images/{i}_encoded.png")
        save_image((encoded_images - undo_norm(images))**2*64, f"../preservation_images/{i}_diff.png")

        encoded_embedding = vit(vit_transform(norm(noised_images)))
        encoded_embedding = torch.nn.functional.normalize(encoded_embedding, p=2, dim=1)
        images_embedding = vit(vit_transform(norm(images)))
        images_embedding = torch.nn.functional.normalize(images_embedding, p=2, dim=1)
        mse = torch.nn.functional.mse_loss(encoded_embedding, images_embedding)

        print(mse)

        val_a += (mse < 0.0005).sum().item()
        val_b += (mse < 0.0002).sum().item()
        tot += mse.numel()

        print(f"Val A: {val_a / tot:.4f}")
        print(f"Val B: {val_b / tot:.4f}")

        processed_image = encoded_images
        diff = torch.abs(message - decoded_messages) > 0.5
        bitwise_error += diff.sum().item() / len(message)

        print(f"Bitwise error rate: {bitwise_error / tot:.4f}")

        continue

        if MODE == "JPEG":
            processed_image = jpeg_compress_tensor_multiple(
                encoded_images, quality=QUALITY
            ).cuda()
            processed_image = norm(processed_image)
            # print(compressed_images[0,0,:5,:5])
            # print(encoded_images[0,0,:5,:5])
            # print((compressed_images - encoded_images).abs().mean())
        elif MODE == "NOISE":
            processed_image = images + torch.randn_like(images) * NOISE_AMOUNT
        else:
            processed_image = encoded_images

        decoded_messages = decoder(processed_image)
        
        diff = torch.abs(message - decoded_messages[0]) > 0.5
        error += diff.sum().item()
        tot += diff.numel()

        if i % 10 == 0:
            print(f"Error rate: {error / tot:.4f}")

    print(f"Final error rate: {error / tot:.4f}")

main()