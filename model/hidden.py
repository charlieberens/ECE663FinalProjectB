import numpy as np
import torch
import torch.nn as nn
import torchvision

from options import HiDDenConfiguration
from model.discriminator import Discriminator
from model.encoder_decoder import EncoderDecoder, SplittyDecoderWrapper
from vgg_loss import VGGLoss
from noise_layers.noiser import Noiser
from PIL import Image
from io import BytesIO

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

def jpeg_compress_tensor(tensor: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """
    Compress a PyTorch tensor using JPEG and return the result as a new tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor with shape (C, H, W) in the range [0, 1].
        quality (int): JPEG compression quality (1-100).
    
    Returns:
        torch.Tensor: JPEG-compressed tensor.
    """
    tensor = tensor.clamp(0, 1)

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

class Hidden:
    def __init__(self, configuration: HiDDenConfiguration, device: torch.device, noiser: Noiser, tb_logger):
        """
        :param configuration: Configuration for the net, such as the size of the input image, number of channels in the intermediate layers, etc.
        :param device: torch.device object, CPU or GPU
        :param noiser: Object representing stacked noise layers.
        :param tb_logger: Optional TensorboardX logger object, if specified -- enables Tensorboard logging
        """
        super(Hidden, self).__init__()

        self.encoder_decoder = EncoderDecoder(configuration, noiser).to(device)
        self.discriminator = Discriminator(configuration).to(device)
        self.optimizer_enc_dec = torch.optim.Adam(self.encoder_decoder.parameters())
        self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters())

        if configuration.use_vgg:
            self.vgg_loss = VGGLoss(3, 1, False)
            self.vgg_loss.to(device)
        else:
            self.vgg_loss = None

        self.config = configuration
        self.device = device

        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)

        # Defined the labels used for training the discriminator/adversarial loss
        self.cover_label = 1
        self.encoded_label = 0

        self.vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT).cuda()
        self.vit_transform = torchvision.transforms.Resize(224)

        # Freeze the VIT model
        for param in self.vit.parameters():
            param.requires_grad = False

        self.norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.tb_logger = tb_logger
        if tb_logger is not None:
            from tensorboard_logger import TensorBoardLogger
            encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            encoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/encoder_out'))
            decoder_final = self.encoder_decoder.decoder._modules['linear']
            decoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/decoder_out'))
            discrim_final = self.discriminator._modules['linear']
            discrim_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/discrim_out'))


    def train_on_batch(self, batch: list):
        """
        Trains the network on a single batch consisting of images and messages
        :param batch: batch of training data, in the form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        images, messages = batch

        batch_size = images.shape[0]
        self.encoder_decoder.train()
        self.discriminator.train()

        DO_SLOW_EVAL = False

        with torch.enable_grad():
            # ---------------- Train the discriminator -----------------------------
            self.optimizer_discrim.zero_grad()
            # train on cover
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover.float())
            d_loss_on_cover.backward()

            # train on fake
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
            d_on_encoded = self.discriminator(encoded_images.detach())
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded.float())


            if DO_SLOW_EVAL:
                unnorm_encoded_images = (encoded_images + 1) / 2

                compressed_images = jpeg_compress_tensor_multiple(unnorm_encoded_images, quality=75).cuda()

                torchvision.utils.save_image((compressed_images * 2) - 1, f"compressed_scaled.png")
                torchvision.utils.save_image(encoded_images, f"encoded_scaled.png")

                compressed_decoded_messages = self.encoder_decoder.decoder(self.encoder_decoder.split_image(compressed_images * 2 -1)).view((images.shape[0], -1))
                compressed_decoded_messages_mse_loss = self.mse_loss(compressed_decoded_messages, messages.float())

                compressed_decoded_rounded = compressed_decoded_messages.detach().cpu().round().clip(0, 1)
                compressed_bitwise_avg_err_split = torch.sum(torch.abs(compressed_decoded_rounded - messages.detach().cpu()), dim=1) / (messages.shape[1])
                compressed_bitwise_avg_err = torch.sum(compressed_bitwise_avg_err_split) / batch_size
                compressed_bitwise_success_rate = torch.sum(compressed_bitwise_avg_err_split < .05) / batch_size
            else:
                compressed_bitwise_success_rate = torch.tensor(1)
                compressed_bitwise_avg_err = torch.tensor(1)
                compressed_decoded_messages_mse_loss = torch.tensor(1)

            # compressed_decoded_messages = self.encoder_decoder.decoder(self.encoder_decoder.split_image(noised_images)).view((images.shape[0], -1))
            # compressed_decoded_messages_mse_loss = self.mse_loss(compressed_decoded_messages, messages.float())

            d_loss_on_encoded.backward()
            self.optimizer_discrim.step()

            # --------------Train the generator (encoder-decoder) ---------------------
            self.optimizer_enc_dec.zero_grad()
            # target label for encoded images should be 'cover', because we want to fool the discriminator
            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc.float(), g_target_label_encoded.float())

            if self.vgg_loss == None:
                g_loss_enc = self.mse_loss(encoded_images, images.float())
            else:
                # vgg_on_cov = self.vgg_loss(images)
                # vgg_on_enc = self.vgg_loss(encoded_images)
                # vgg_on_enc = self.vgg_loss(noised_images)
                # g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc.float())
                g_loss_enc = self.mse_loss(images, encoded_images)

            encoded_embedding = self.vit(self.vit_transform(noised_images))
            encoded_embedding = torch.nn.functional.normalize(encoded_embedding, p=2, dim=1)
            images_embedding = self.vit(self.vit_transform(images))
            images_embedding = torch.nn.functional.normalize(images_embedding, p=2, dim=1)
            split_vit_loss = torch.nn.functional.mse_loss(encoded_embedding, images_embedding, reduction="none").mean(dim=1)

            if DO_SLOW_EVAL:
                compressed_images_embedding = self.vit(self.vit_transform(self.norm(compressed_images)))
                compressed_images_embedding = torch.nn.functional.normalize(compressed_images_embedding, p=2, dim=1)
                split_compressed_vit_loss = torch.nn.functional.mse_loss(compressed_images_embedding, images_embedding, reduction="none").mean(dim=1)
                compressed_count_fine_a = (split_compressed_vit_loss < .00035).sum()
                compressed_vit_loss = torch.mean(split_compressed_vit_loss)
            else:
                compressed_vit_loss = 0
                compressed_count_fine_a = 1

            vit_count_fine_a = (split_vit_loss < .00035).sum()
            vit_loss = torch.mean(split_vit_loss)

            # Choose a random number
            rand = np.random.rand()

            # if rand > .9:
            if DO_SLOW_EVAL:
                weaved_noised_images_and_images = torch.cat([
                    (encoded_images + 1) / 2,
                    (noised_images + 1) / 2,
                    (images + 1) / 2,
                    compressed_images], dim=3)
                # Save weaved_noised_images_and_images
                torchvision.utils.save_image(weaved_noised_images_and_images, f"weaved_noised_images_and_images-proper-vit-5.png")


            g_loss_dec = self.mse_loss(decoded_messages, messages.float())
            g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
                     + self.config.decoder_loss * g_loss_dec \
                     + .7 * vit_loss 

            g_loss.backward()
            self.optimizer_enc_dec.step()

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item(),
            "vit_loss": vit_loss.item(),
            "compressed_vit_loss": compressed_vit_loss,
            "vit_count_fine_a": vit_count_fine_a / batch_size,
            "compressed_count_fine_a": compressed_count_fine_a / batch_size,
            "compressed_decoded_messages_mse_loss": compressed_decoded_messages_mse_loss,
            "compressed_bitwise_avg_err": compressed_bitwise_avg_err,
            "compressed_bitwise_success_rate": compressed_bitwise_success_rate
        }
        return losses, (encoded_images, noised_images, decoded_messages)

    def validate_on_batch(self, batch: list):
        """
        Runs validation on a single batch of data consisting of images and messages
        :param batch: batch of validation data, in form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        # if TensorboardX logging is enabled, save some of the tensors.
        if self.tb_logger is not None:
            encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            self.tb_logger.add_tensor('weights/encoder_out', encoder_final.weight)
            decoder_final = self.encoder_decoder.decoder._modules['linear']
            self.tb_logger.add_tensor('weights/decoder_out', decoder_final.weight)
            discrim_final = self.discriminator._modules['linear']
            self.tb_logger.add_tensor('weights/discrim_out', discrim_final.weight)

        images, messages = batch

        batch_size = images.shape[0]

        self.encoder_decoder.eval()
        self.discriminator.eval()
        with torch.no_grad():
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover.float())

            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

            d_on_encoded = self.discriminator(encoded_images)
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded.float())

            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded.float())

            if self.vgg_loss is None:
                g_loss_enc = self.mse_loss(encoded_images, images)
            else:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_images)
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

            g_loss_dec = self.mse_loss(decoded_messages, messages)
            g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
                     + self.config.decoder_loss * g_loss_dec

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item()
        }
        return losses, (encoded_images, noised_images, decoded_messages)

    def to_stirng(self):
        return '{}\n{}'.format(str(self.encoder_decoder), str(self.discriminator))
