from io import BytesIO
import numpy as np
from torchvision import datasets, transforms
import torchvision
import torch
import argparse
from PIL import Image
from torchvision.utils import save_image
import matplotlib.pyplot as plt


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    args = parser.parse_args()

    # vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
    vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
    fcl = torch.nn.Linear(1000, 96)
    # softmax = torch.nn.Softmax(dim=1)

    torch.nn.init.xavier_uniform_(fcl.weight)
    torch.nn.init.zeros_(fcl.bias)

    # model = torch.nn.Sequential(vit, fcl)
    model = torch.nn.Sequential(vit)

    train_images = datasets.ImageFolder(args.data_path, transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop((224, 224), pad_if_needed=True),
            transforms.ToTensor(),
        ]))
    
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=4, shuffle=True,
                                               num_workers=2)

    compresion_levels = [95, 90, 80]
    noise_magnitude = 0.01
    noise_magnitudes = [0.01, 0.02, 0.05]

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    l_matching = 0
    l_different = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model = model.to("cuda")
    model.train()

    compression_distances = [[] for _ in compresion_levels]
    noise_distances = [[] for _ in noise_magnitudes]
    different_distances = []
    for k in range(8):
        for j, (image, _) in enumerate(train_loader):
            if(len(image) < 4):
                continue
            a_images = [image[0]]
            b = image[1:]
            for quality in compresion_levels:
                a_images.append(jpeg_compress_tensor(image[0], quality))
            for noise in noise_magnitudes:
                a_images.append(image[0] + torch.randn_like(image[0]) * noise)
            
            a_images = torch.stack(a_images)
            b = b
            all_images = torch.cat([a_images, b], dim=0)

            # Remove the last 4 bits from the iamge
            all_images = (all_images * 255).int() / 16
            all_images = all_images.float() / 16

            # Normalize images
            all_images = norm(all_images)
            all_images = all_images.to("cuda")

            # Forward pass
            output = model(all_images)

            # Normalize output
            output = torch.nn.functional.normalize(output, p=2, dim=1)

            # Compute similarity
            a = output[0]
            compressed_ones = output[1:len(compresion_levels)+1]
            noisy_ones = output[len(compresion_levels)+1:len(compresion_levels)+1+len(noise_magnitudes)]
            b = output[len(compresion_levels)+1+len(noise_magnitudes):]

            for i in range(len(compresion_levels)):
                compression_distances[i].append(torch.nn.functional.mse_loss(a, compressed_ones[i]).item())
            
            for i in range(len(noise_magnitudes)):
                noise_distances[i].append(torch.nn.functional.mse_loss(a, noisy_ones[i]).item())
            
            for i in range(len(b)):
                different_distances.append(torch.nn.functional.mse_loss(a, b[i]).item())

            if j % 100 == 0:
                print(f"{j}/{len(train_loader)}")

    obj_of_the_things = {
        "compression_distances": compression_distances,
        "noise_distances": noise_distances,
        "different_distances": different_distances
    }

    torch.save(obj_of_the_things, "temp.pth")

    # Plot the distribution of different_distances
    plt.hist(different_distances, bins=18, alpha=0.75, range=[0, 0.003], label="Different Images", color="#9c7de3")
    orange_colors = ["#e3847d", "#e3ae7d", "#e3cb7d"]
    green_colors = ["#9fe37d", "#7de3a1", "#7de3e0"]
    for i in range(len(noise_magnitudes)):
        # Plot the distribution of noise_distances[i] all different shades of green
        plt.hist(noise_distances[i], bins=18, alpha=.75, range=[0, 0.003], label=f"Noise ({noise_magnitudes[i]})", color=green_colors[i])
    
    plt.legend()
    plt.savefig("temp1.png")

    # Clear the plot
    plt.clf()

    plt.hist(different_distances, bins=18, alpha=0.75, range=[0, 0.003], label="Different Images", color="#9c7de3")
    for i in range(len(compresion_levels)):
        # Plot the distribution of compression_distances[i] all different shades of orange
        plt.hist(compression_distances[i], bins=18, alpha=0.75, range=[0, 0.003], label=f"Compression ({compresion_levels[i]})", color=orange_colors[i])
        
    plt.legend()
    # Save the plot
    plt.savefig("temp2.png")

    exit()
    for epoch in range(100):
        total_similar = 0
        total_different = 0

        for j, (image, _) in enumerate(train_loader):
            if(len(image) != 2):
                continue
            a_images = [image[0]]
            b_images = [image[1]]
            for quality in compresion_levels:
                a_images.append(jpeg_compress_tensor(image[0], quality))
                b_images.append(jpeg_compress_tensor(image[1], quality))
            
            a_images = torch.stack(a_images)
            b_images = torch.stack(b_images)
            all_images = torch.cat([a_images, b_images], dim=0)

            # Remove the last 4 bits from the iamge
            all_images = (all_images * 255).int() / 16
            all_images = all_images.float() / 16

            # Save all images
            # save_image(all_images, f"temp.jpg")
            
            all_images = norm(all_images)
            all_images = all_images.to("cuda")

            # Add gaussian noise
            all_images += torch.randn_like(all_images) * noise_magnitude

            # Forward pass
            output = model(all_images)

            # Normalize output
            output = torch.nn.functional.normalize(output, p=2, dim=1)

            # Compute similarity
            similar = 0
            different = torch.nn.functional.mse_loss(
                output[len(compresion_levels)+1].unsqueeze(0), output[0].unsqueeze(0)
            )

            for i in range(len(compresion_levels)):
                a = output[i]
                b = output[i + len(compresion_levels)+2]

                matching_diff_a = torch.nn.functional.mse_loss(a.unsqueeze(0), output[0].unsqueeze(0))
                matching_diff_b = torch.nn.functional.mse_loss(
                    b.unsqueeze(0), output[len(compresion_levels)+1].unsqueeze(0)
                )
                matching_diff = matching_diff_a + matching_diff_b

                different_diff = torch.nn.functional.mse_loss(
                    b.unsqueeze(0), output[0].unsqueeze(0)
                ) + torch.nn.functional.mse_loss(
                    a.unsqueeze(0), output[len(compresion_levels)+1].unsqueeze(0)
                )

                similar += matching_diff
                different += different_diff
            
            different = different / (2*len(compresion_levels) + 1)
            similar = similar / (2*len(compresion_levels))

            loss = l_matching * similar - l_different * different

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_similar += similar.item()
            total_different += different.item()

            if j % 100 == 0 or True:
                print(f"[{epoch}:{j+1}] Similar: {float(similar)}, Opposite: {float(different)}")
        
        print(f"Epoch {epoch}, Mean Similarity: {total_similar / (len(train_loader) * compresion_levels)}, Mean Opposite: {total_different / (len(train_loader) * compresion_levels)}")

main()