import math
import torch
import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu
from torchvision import transforms

class Decoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(Decoder, self).__init__()
        self.channels = config.decoder_channels

        if config.split_image_into_16x16_blocks:
            message_length = config.message_block_length
            self.H = 16
            self.W = 16
            self.mask = None
        else:
            message_length = config.message_length
            self.H = config.H
            self.W = config.W
            self.mask = config.mask.cuda()

        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(config.decoder_blocks - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))

        # layers.append(block_builder(self.channels, config.message_length))
        layers.append(ConvBNRelu(self.channels, message_length))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(message_length, message_length)

    def forward(self, image_with_wm):
        if self.mask != None:
            x = self.mask * image_with_wm
        x = self.layers(image_with_wm)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        x.squeeze_(3).squeeze_(2)
        x = self.linear(x)
        return x

class BitwiseDecoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, config: HiDDenConfiguration):

        super(BitwiseDecoder, self).__init__()
        self.channels = config.decoder_channels

        if config.split_image_into_16x16_blocks:
            message_length = config.message_block_length
            self.H = 16
            self.W = 16
            self.mask = None
        else:
            message_length = config.message_length
            self.H = config.H
            self.W = config.W
            self.mask = config.mask.cuda()

        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(config.decoder_blocks - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))

        # layers.append(block_builder(self.channels, config.message_length))
        layers.append(ConvBNRelu(self.channels, message_length))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(message_length, message_length)
        self.num_bits = int(config.masking_args)
        self.int_rounding_factor = 2**self.num_bits

        # We take the bits to be uniformly distributed on the interval [0,1]. Therefore, their mean should be .5, and their standard deviation should be sqrt((b-a)^2/12) = (sqrt(1/12))
        root_one_twelveth = 1/math.sqrt(12)
        self.bit_norm = transforms.Normalize(
            [.5, .5, .5],
            [root_one_twelveth, root_one_twelveth, root_one_twelveth]
        )

    def forward(self, image_with_wm):
        if self.mask != None:
            x = self.mask * image_with_wm
        
        rounded_down_image = torch.floor(
            image_with_wm * 256 / self.int_rounding_factor) * self.int_rounding_factor / 256
        
        last_few_bits = image_with_wm - rounded_down_image
        last_few_bits = self.bit_norm(last_few_bits)

        x = self.layers(last_few_bits)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        x.squeeze_(3).squeeze_(2)
        x = self.linear(x)
        return x

