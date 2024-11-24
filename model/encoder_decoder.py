import torch.nn as nn
from model.encoder import BitwiseEncoder, BitwiseEncoder2, Encoder
from model.decoder import BitwiseDecoder, Decoder
from options import HiDDenConfiguration
from noise_layers.noiser import Noiser


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
        # self.decoder = Decoder(config)
        self.decoder = BitwiseDecoder(config)

    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        noised_and_cover = self.noiser([encoded_image, image])
        noised_image = noised_and_cover[0]
        decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message
