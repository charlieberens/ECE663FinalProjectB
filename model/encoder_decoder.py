import torch
import torch.nn as nn
from model.encoder import BitwiseEncoder, BitwiseEncoder2, Encoder
from model.decoder import BitwiseDecoder, Decoder
from options import HiDDenConfiguration
from noise_layers.noiser import Noiser
from torchvision.utils import save_image

class EncoderDecoder(nn.Module):
    """
    Combines Encoder->Noiser->Decoder into single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a three-tuple: (encoded_image, noised_image, decoded_message)
    """
    def __init__(self, config: HiDDenConfiguration, noiser: Noiser):

        super(EncoderDecoder, self).__init__()

        if config.hash_mode == "bitwiseA":
            self.encoder = BitwiseEncoder(config)
        elif config.hash_mode == "bitwiseB":
            self.encoder = BitwiseEncoder2(config)
        elif config.hash_mode == "bitwiseC":
            self.encoder = BitwiseEncoder2(config, include_image=True)
        else:
            self.encoder = Encoder(config)

        self.noiser = noiser
        self.decoder = Decoder(config)
        if config.hash_mode == "bitwiseA":
            self.decoder = BitwiseDecoder(config)
        elif config.hash_mode == "bitwiseB":
            self.decoder = BitwiseDecoder(config)
        elif config.hash_mode == "bitwiseC":
            self.decoder = BitwiseDecoder(config)
        else:
            self.decoder = Decoder(config)

        self.split_image_into_16x16_blocks = config.split_image_into_16x16_blocks
        self.message_block_length = config.message_block_length
        self.batch_size = config.batch_size
        self.H = config.H
        self.W = config.W
        self.split_amount = self.H // 16

    def split_image(self, image):
        # Split image
        image = torch.chunk(image, self.split_amount, dim=2)
        image = [torch.chunk(i, self.split_amount, dim=3) for i in image]
        image = tuple(item for subtuple in image for item in subtuple)
        image = torch.stack(image)
        image = image.permute(1,0,2,3,4).reshape((-1, image.shape[2], image.shape[3], image.shape[4]))
        return image

    def unsplit_image(self, image, current_batch_size):
        image = image.reshape((current_batch_size, -1, image.shape[1], image.shape[2], image.shape[3]))
        image = image.split(1, dim=1)
        image = torch.cat([torch.cat(image[n*self.split_amount:(n+1)*self.split_amount], dim=4).squeeze() for n in range(len(image) // self.split_amount)], dim=2)
        return image

    def forward(self, image, message):
        if self.split_image_into_16x16_blocks:
            current_batch_size = image.shape[0]
            image = self.split_image(image)
            message = message.view((message.shape[0] * (message.shape[1] // self.message_block_length),self.message_block_length))
            encoded_image = self.encoder(image, message)
            encoded_image = self.unsplit_image(encoded_image, current_batch_size)
            noised_and_cover = self.noiser([encoded_image, image])
            noised_image = noised_and_cover[0]
            
            # Add gaussian noise to the image
            noised_image = noised_image + torch.randn_like(noised_image) * 0.1

            split_noised_image = self.split_image(noised_image)
            decoded_message = self.decoder(split_noised_image)
            decoded_message = decoded_message.view((current_batch_size, -1))
        else:
            encoded_image = self.encoder(image, message)
            noised_and_cover = self.noiser([encoded_image, image])
            noised_image = noised_and_cover[0]
            decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message

class SplittyDecoderWrapper(nn.Module):
    def __init__(self,encoder_decoder: EncoderDecoder, config: HiDDenConfiguration):
        super(SplittyDecoderWrapper, self).__init__()

        self.decoder = encoder_decoder.decoder
        self.encoder_decoder = encoder_decoder

        self.split_image_into_16x16_blocks = config.split_image_into_16x16_blocks
        self.message_block_length = config.message_block_length
        self.batch_size = config.batch_size
        self.H = config.H
        self.W = config.W
        self.split_amount = self.H // 16

    def forward(self, image):
        split_noised_image = self.encoder_decoder.split_image(image)
        decoded_message = self.decoder(split_noised_image)
        decoded_message = decoded_message.view((image.shape[0], -1))

        return decoded_message