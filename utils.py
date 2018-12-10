import numpy as np
import os
import re
import csv
import time
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils import data
from options import HiDDenConfiguration, TrainingOptions
from model.hidden import Hidden
import pickle


def image_to_tensor(image):
    """
    Transforms a numpy-image into torch tensor
    :param image: (batch_size x height x width x channels) uint8 array
    :return: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    """
    image_tensor = torch.Tensor(image)
    image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.permute(0, 3, 1, 2)
    image_tensor = image_tensor / 127.5 - 1
    return image_tensor


def tensor_to_image(tensor):
    """
    Transforms a torch tensor into numpy uint8 array (image)
    :param tensor: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    :return: (batch_size x height x width x channels) uint8 array
    """
    image = tensor.permute(0, 2, 3, 1).cpu().numpy()
    image = (image + 1) * 127.5
    return np.clip(image, 0, 255).astype(np.uint8)


def show_images(images, labels):
    f, axarr = plt.subplots(1, len(images))
    for i in range(len(images)):
        axarr[i].imshow(images[i])
        axarr[i].set_title(labels[i])
    plt.show()


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def last_checkpoint_from_folder(folder: str):
    last_file = sorted_nicely(os.listdir(folder))[-1]
    last_file = os.path.join(folder, last_file)
    return last_file


def save_checkpoint(model: Hidden, epoch: int, losses_accu: dict, checkpoint_folder: str, file_name_prefix=''):
    """ Saves a checkpoint at the end of an epoch. """
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    losses_redux = {name.strip(): np.mean(value) for (name, value) in losses_accu.items()}
    checkpoint_filename = '{}{}-loss={:.4f} enc={:.4f} dec={:.4f} adv={:.4f} discr-cov={:.4f} discr-enc={:.4f}.pyt'.\
        format(
            epoch,
            file_name_prefix,
            losses_redux['loss'],
            losses_redux['encoder_mse'],
            losses_redux['dec_mse'],
            losses_redux['adversarial_bce'],
            losses_redux['discr_cover_bce'],
            losses_redux['discr_encod_bce']
    )
    checkpoint_filename = os.path.join(checkpoint_folder, checkpoint_filename)
    print('Saving checkpoint to {}'.format(checkpoint_filename))
    checkpoint = {
        'enc-dec-model': model.encoder_decoder.state_dict(),
        'enc-dec-optim': model.optimizer_enc_dec.state_dict(),
        'discrim-model': model.discriminator.state_dict(),
        'discrim-optim': model.optimizer_discrim.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_filename)
    print('Saving checkpoint done.')


# def load_checkpoint(hidden_net: Hidden, options: Options, this_run_folder: str):
def load_last_checkpoint(checkpoint_folder):
    """ Load the last checkpoint from the given folder """
    last_checkpoint_file = last_checkpoint_from_folder(checkpoint_folder)
    print("=> loading checkpoint '{}'".format(last_checkpoint_file))
    checkpoint = torch.load(last_checkpoint_file)
    print("=> loaded checkpoint '{}' (epoch {})".format(last_checkpoint_file, checkpoint['epoch']))

    return checkpoint


def model_from_checkpoint(hidden_net, checkpoint):
    """ Restores the hidden_net object from a checkpoint object """
    hidden_net.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])
    hidden_net.optimizer_enc_dec.load_state_dict(checkpoint['enc-dec-optim'])
    hidden_net.discriminator.load_state_dict(checkpoint['discrim-model'])
    hidden_net.optimizer_discrim.load_state_dict(checkpoint['discrim-optim'])


def load_options(run_folder, options_file_name = 'options-and-config.pickle'):
    """ Loads the training, model, and noise configurations from the given folder """
    with open(os.path.join(run_folder, options_file_name), 'rb') as f:
        train_options = pickle.load(f)
        noise_config = pickle.load(f)
        hidden_config = pickle.load(f)

    return train_options, hidden_config, noise_config



def get_data_loaders(hidden_config: HiDDenConfiguration, train_options: TrainingOptions):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.CenterCrop((hidden_config.H, hidden_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((hidden_config.H, hidden_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_images = datasets.ImageFolder(train_options.train_folder, data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=train_options.batch_size, shuffle=True, num_workers=4)

    validation_images = datasets.ImageFolder(train_options.validation_folder, data_transforms['test'])
    validation_loader = torch.utils.data.DataLoader(validation_images, batch_size=train_options.batch_size, shuffle=False, num_workers=4)

    return train_loader, validation_loader

def print_progress(losses_accu):
    max_len = max([len(loss_name) for loss_name in losses_accu])
    for loss_name, loss_value in losses_accu.items():
        print(loss_name.ljust(max_len+4) + '{:.4f}'.format(np.mean(loss_value)))


def create_folder_for_run(options: TrainingOptions):
    if not os.path.exists(options.runs_folder):
        os.makedirs(options.runs_folder)

    this_run_folder = time.strftime("%Y.%m.%d--%H-%M-%S")

    this_run_folder = os.path.join(options.runs_folder, this_run_folder)
    os.makedirs(this_run_folder)
    os.makedirs(os.path.join(this_run_folder, 'checkpoints'))

    return this_run_folder


def write_losses(file_name, losses_accu, epoch, duration):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()] + ['duration']
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(np.mean(loss_list)) for loss_list in losses_accu.values()] + ['{:.0f}'.format(duration)]
        writer.writerow(row_to_write)