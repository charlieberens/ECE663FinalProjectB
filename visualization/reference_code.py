import os

import torch
import torchvision

IMG_PATH = "/usr/xtmp/cjb131/output_images/3"

original_image_path = os.path.join(IMG_PATH, "original")

variants = [
    "signed",
    "signed_compressed_100",
    "signed_compressed_90",
    "signed_compressed_75",
    "unsigned_compressed_100",
    "unsigned_compressed_90",
    "unsigned_compressed_75",
]

variant_paths = [os.path.join(IMG_PATH, variant) for variant in variants]

def main():
    # Initialize ViT model
    vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT).cuda()
    vit_transform = torchvision.transforms.Resize(224)

    norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # NOTE: Files are named 0,1,2,...,n.pkl
    for file in os.listdir(original_image_path):
        # Load each variant, and combine them into a single tensor
        variant_tensors = [vit_transform(norm(torch.load(os.path.join(variant_path, file)))) for variant_path in [original_image_path] + variant_paths]
        stacked = torch.stack(variant_tensors).cuda()

        # Get the ViT embeddings for each variant
        output = vit(stacked)
        normalized_output = torch.nn.functional.normalize(output, p=2, dim=1)

        # Split the embeddings
        og_image_embedding = normalized_output[0]
        variant_embeddings = normalized_output[1:]

        # Calculate the L2 Dist between the original image and each variant, why do I do it this way? Don't worry about it.
        l2s = [torch.sqrt(torch.nn.functional.mse_loss(og_image_embedding, variant_embedding) * 1000) for variant_embedding in variant_embeddings]

        for variant,l2 in zip(variants, l2s):
            print(f"{variant}: {l2.item():.8f}")
        print()

if __name__ == "__main__":
    main()