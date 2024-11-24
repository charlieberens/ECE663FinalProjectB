from model.encoder import BitwiseEncoder, BitwiseEncoder2
from PIL import Image
from options import HiDDenConfiguration
from torchvision import transforms
import torchvision
import torch

from datasets import load_dataset

def main():
    im = Image.open("../data/coco_fixed_small/train/a/img077304.jpg")
    to_tensor = transforms.Compose([
                transforms.CenterCrop((200, 200)),
                transforms.ToTensor(),
            ])
    im_tensor = to_tensor(im).unsqueeze(0)

    hidden_config = HiDDenConfiguration(H=200, W=200,
                                        message_length=32,
                                        encoder_blocks=4, encoder_channels=64,
                                        decoder_blocks=7, decoder_channels=64,
                                        use_discriminator=True,
                                        use_vgg=False,
                                        discriminator_blocks=3, discriminator_channels=64,
                                        decoder_loss=1,
                                        encoder_loss=0.7,
                                        adversarial_loss=1e-3,
                                        enable_fp16=True,
                                        mask=None,
                                        masking_args=8,
                                        hash_mode="bitwise",
    )

    encoder = BitwiseEncoder2(hidden_config)
    enc = encoder(im_tensor, torch.zeros((1,32)))
    torchvision.utils.save_image(im_tensor, "temp.jpg")
    # print(im_tensor)
    # print(enc)
    # print(im_tensor.min(),im_tensor.max())
    a = encoder.to_int_tensor(enc)
    b = encoder.to_int_tensor(im_tensor)
    print(b)
    print(a)
    print((a.int()-b.int()).max())
    print((a.int()-b.int()).min())
    print((a.float()-b.float()).mean())
main()