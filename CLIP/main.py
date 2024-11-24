import torch
import clip
from PIL import Image
from torchvision import datasets, transforms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("train_folder", type=str)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

data_transforms = {
    'train': torch.nn.Sequential([
        transforms.Resize(512),
        transforms.CenterCrop((512, 512)),
        preprocess,
        transforms.ToTensor(),
    ]),
    'test': torch.nn.Sequential([
        transforms.Resize(512),
        transforms.CenterCrop((512, 512)),
        preprocess,
        transforms.ToTensor(),
    ])
}
train_images = datasets.ImageFolder(args.train_folder, data_transforms['train'])
train_loader = torch.utils.data.DataLoader(train_images, batch_size=12, shuffle=True,
                                            num_workers=2)

with torch.no_grad():
    for image in train_loader:
        image = image.to(device)
        image_features = model.encode_image(image)

        print(image_features)

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]