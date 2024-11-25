from io import BytesIO
import os
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
    args = parser.parse_args()

    norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    undo_norm = transforms.Normalize([-1, -1, -1], [2, 2, 2])

    train_images = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop((256, 256), pad_if_needed=True),
        transforms.ToTensor(),
        # norm
    ]))
    val_images = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        # norm
    ]))
    
    vit_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop((224, 224), pad_if_needed=True),
            transforms.ToTensor(),
        ])
    vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
    
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=8, shuffle=True,
                                               num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_images, batch_size=8, shuffle=False, 
                                                num_workers=2)

    # Set up all the hasing we're doing
    aes_key = os.urandom(32)
    iv = os.urandom(16)

    # Print key and iv as hex strings
    print(f"Key: {aes_key.hex()}")
    print(f"IV: {iv.hex()}")

    cipher = Cipher(algorithms.AES(aes_key), modes.CFB(iv))
    encryptor = cipher.encryptor()
    decryptor = cipher.decryptor()

    # Set up an ed25519 keypair
    # This isn't a typo. We are broadcasting, so we want to encrypt with the private key and decrypt with the public key.
    # To do this in the cryptography we gotta do this nonsense.
    public_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    private_key = public_key.public_key()

    print(f"Private key: {private_key.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)}")
    print(f"Public key: {public_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())}")

    # Concatenate the AES key, with the IV, with a string "GIBBON"
    concatenated = aes_key + iv + b"GIBBON"

    # Encrypt the concatenated string using the private key
    encrypted_aes_key = private_key.encrypt(
        concatenated,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # Decrypt the concatenated string using the public key
    decrypted_aes_key = public_key.decrypt(
        encrypted_aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    print(f"Decrypted AES key: {decrypted_aes_key}")

    # Set up the encoder decoder
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

    error = 0
    tot = 0

    MODE = "JPEG"
    QUALITY = 100

    if MODE == "JPEG":
        print(f"Using JPEG compression with quality {QUALITY}")

    for i, (images, _) in enumerate(val_loader):
        # pre_vit = vit_transform(images)
        # pre_vit = pre_vit.to(device)
        # pre_vit = vit(pre_vit)

        images = images.to(device)
        old_images = images
        images = norm(images)

        messages = message.unsqueeze(0).repeat(images.shape[0], 1)
        encoded_images = model.encoder_decoder(images, messages)[0]
        encoded_images = undo_norm(encoded_images)

        if MODE == "JPEG":
            compressed_images = jpeg_compress_tensor_multiple(
                encoded_images, quality=QUALITY
            ).cuda()
            compressed_images = norm(compressed_images)
            # print(compressed_images[0,0,:5,:5])
            # print(encoded_images[0,0,:5,:5])
            # print((compressed_images - encoded_images).abs().mean())

        decoded_messages = decoder(compressed_images)
        
        diff = torch.abs(message - decoded_messages[0]) > 0.5
        error += diff.sum().item()
        tot += diff.numel()

        if i % 10 == 0:
            print(f"Error rate: {error / tot:.4f}")

    print(f"Final error rate: {error / tot:.4f}")

main()