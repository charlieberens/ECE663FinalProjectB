import torch
import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu
from torchvision import datasets, transforms


class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(Encoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        self.mask = config.mask.cuda()

        layers = [ConvBNRelu(3, self.conv_channels)]

        for _ in range(config.encoder_blocks-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + config.message_length,
                                             self.conv_channels)

        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)


    def forward(self, image, message):

        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)

        expanded_message = expanded_message.expand(-1,-1, self.H, self.W)
        encoded_image = self.conv_layers(image)
        # concatenate expanded message and image
        concat = torch.cat([expanded_message, encoded_image, image], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)

        if self.mask != None:
            im_w = self.mask * im_w

        return im_w

class BitwiseEncoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(BitwiseEncoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        layers = [ConvBNRelu(3, self.conv_channels)]

        for _ in range(config.encoder_blocks-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + config.message_length,
                                             self.conv_channels)
        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)
        self.num_bits = int(config.masking_args)
        self.int_rounding_factor = 2**self.num_bits

        self.norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        self.output_scaling_factor = ((self.int_rounding_factor-1)/256)
        self.bit_change_temp = 10

    def forward(self, image, message):
        # NOTE - This image must be unnormalized
        # Set the last self.num_bits to 0
        pre_norm_image = torch.floor(image * 256 / self.int_rounding_factor) * self.int_rounding_factor / 256
        image = self.norm(pre_norm_image)

        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)

        expanded_message = expanded_message.expand(-1,-1, self.H, self.W)
        encoded_image = self.conv_layers(image)
        # # concatenate expanded message and image
        concat = torch.cat([expanded_message, encoded_image, image], dim=1)
        im_w_diff_val = self.after_concat_layer(concat)
        im_w_diff_val = self.final_layer(im_w_diff_val)

        im_w_diff_val = im_w_diff_val - image

        im_w_diff = torch.nn.functional.sigmoid(self.bit_change_temp * im_w_diff_val)

        im_w = pre_norm_image + self.output_scaling_factor * im_w_diff
        im_w = self.norm(im_w)

        return im_w
    
    @staticmethod
    def to_int_tensor(image):
        return (image * 256).floor().to(torch.uint8)
    
class BitwiseEncoder2(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, config: HiDDenConfiguration, include_image=False):
        super(BitwiseEncoder2, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        layers = [ConvBNRelu(3, self.conv_channels)]

        for _ in range(config.encoder_blocks-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        if include_image:
            self.after_concat_layer = ConvBNRelu(
                self.conv_channels + 6 + config.message_length,
                                                self.conv_channels)
        else:
            self.after_concat_layer = ConvBNRelu(
                self.conv_channels + 3 + config.message_length,
                                                self.conv_channels)
        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)
        self.num_bits = int(config.masking_args)
        self.int_rounding_factor = 2**self.num_bits

        self.output_scaling_factor = ((self.int_rounding_factor-1)/256)
        self.bit_change_temp = 10
        self.include_image = include_image

        self.bit_norm = transforms.Normalize(
            [0.5 * 1/self.int_rounding_factor, 0.5 * 1/self.int_rounding_factor, 0.5 * 1/self.int_rounding_factor], [0.5 * 1/self.int_rounding_factor, 0.5 * 1/self.int_rounding_factor, 0.5 * 1/self.int_rounding_factor]
        )
        self.img_norm = transforms.Normalize(
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        )

    def forward(self, image, message):
        # NOTE - This image must be unnormalized
        # Set the last self.num_bits to 0
        rounded_down_image = torch.floor(image * 256 / self.int_rounding_factor) * self.int_rounding_factor / 256
        last_few_bits = image - rounded_down_image

        if self.include_image:
            normed_image = self.img_norm(image)
        # image = self.norm(pre_norm_image)

        # last_few_bits = self.bit_norm(last_few_bits)

        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)

        expanded_message = expanded_message.expand(-1,-1, self.H, self.W)
        encoded_image = self.conv_layers(last_few_bits)
        # # concatenate expanded message and image
        if self.include_image:
            concat = torch.cat([expanded_message, encoded_image, normed_image, last_few_bits], dim=1)
        else:
            concat = torch.cat([expanded_message, encoded_image, last_few_bits], dim=1)
        im_w_diff_val = self.after_concat_layer(concat)
        im_w_diff_val = self.final_layer(im_w_diff_val)

        im_w_diff = torch.nn.functional.sigmoid(self.bit_change_temp * im_w_diff_val)

        im_w = rounded_down_image + self.output_scaling_factor * im_w_diff

        return im_w
    
    @staticmethod
    def to_int_tensor(image):
        return (image * 256).floor().to(torch.uint8)