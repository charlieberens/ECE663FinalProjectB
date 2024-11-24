from PIL import Image
from torchvision import transforms
import torchvision
import torch

from datasets import load_dataset

def to_int_tensor(image):
    return (image * 256).floor().to(torch.uint8)

def main():
    im1 = Image.open("compressed_images/compressed_quality_100.jpg")
    im2 = Image.open("compressed_images/compressed_quality_90.jpg")

    to_tensor = transforms.Compose([
        transforms.CenterCrop((200, 200)),
        transforms.ToTensor(),
    ])

    im1_tensor = to_tensor(im1).unsqueeze(0)
    im2_tensor = to_tensor(im2).unsqueeze(0)

    a = to_int_tensor(im1_tensor)
    b = to_int_tensor(im2_tensor)

    a_chopped = torch.bitwise_and(a, 255-15)
    b_chopped = torch.bitwise_and(b, 255-15)

    # print(a_chopped - b_chopped)
    print(a_chopped[0,:5,:5])
    print(b_chopped[0,:5,:5])

    print(torch.tensor((a_chopped - b_chopped) > 0, dtype=torch.float32).sum())
main()