import argparse
import math
import os
import sys

import numpy as np
import torch
from PIL import Image
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.base import Callback
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from torch import nn
from torch.autograd import Variable
from torch.nn.init import xavier_uniform
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision import transforms


def mnist_data_loader(args):
    # Create DataLoader for MNIST
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    train_loader = DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_loader


def initializer(m):
    # Run xavier on all weights and zero all biases
    if hasattr(m, 'weight'):
        if m.weight.ndimension() > 1:
            xavier_uniform(m.weight.data)
    if hasattr(m, 'bias'):
        m.bias.data.zero_()


def format_images(images):
    # convert (n, c, h, w) to a single image grid (1, c, g*h, g*w)
    c = images.size(1)
    h = images.size(2)
    w = images.size(3)
    gridsize = int(math.floor(math.sqrt(images.size(0))))
    images = images[:gridsize * gridsize]  # (g*g, c, h, w)
    images = images.view(gridsize, gridsize, c, h, w)  # (g,g,c,h,w)
    images = images.permute(0, 3, 1, 4, 2).contiguous()  # (g, h, g, w, c)
    images = images.view(1, gridsize * h, gridsize * w, c)  # (1, g*h, g*w, c)
    images = images.permute(0, 3, 1, 2)  # (1, c, g*h, g*w)
    return images


# Command line to make images into a video
FFMPEG = """
ffmpeg -r 60 -f image2 -s 280x280 -i \
"generated_images{}%08d.png" -vcodec \
libx264 -crf 25 -pix_fmt yuv420p generation.mp4""".format(os.sep)


def generate_video(path):
    # Run FFMPEG to generate video
    cwd = os.getcwd()
    os.chdir(path=os.path.abspath(path))
    os.system(FFMPEG)
    os.chdir(os.path.abspath(cwd))


def save_args(args):
    # Save argparse arguments to a file for reference
    os.makedirs(args.save_directory, exist_ok=True)
    with open(os.path.join(args.save_directory, 'args.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write("{}={}\n".format(k, v))


class Reshape(nn.Module):
    # Module that just reshapes the input
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class GeneratorNetwork(nn.Sequential):
    # Network for generation
    # Input is (N, latent_dim)
    def __init__(self, args):
        super(GeneratorNetwork, self).__init__(*[m for m in [
            nn.Linear(args.latent_dim, 1024),
            nn.BatchNorm1d(1024) if args.generator_batchnorm else None,
            nn.LeakyReLU(),
            nn.Linear(1024, 7 * 7 * 128),
            Reshape(-1, 128, 7, 7),  # N, 128,7,7
            nn.BatchNorm2d(128) if args.generator_batchnorm else None,
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # N, 64,14,14
            nn.BatchNorm2d(64) if args.generator_batchnorm else None,
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # N, 32,28,28
            nn.BatchNorm2d(32) if args.generator_batchnorm else None,
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),  # N, 1,28,28
            nn.Sigmoid()] if m is not None])


class DiscriminatorNetwork(nn.Sequential):
    # Network for discrimination
    # Input is (N, 1, 28, 28)
    def __init__(self, args):
        super(DiscriminatorNetwork, self).__init__(*[m for m in [
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # N, 64, 14, 14
            nn.BatchNorm2d(64) if args.discriminator_batchnorm else None,
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # N, 128, 7, 7
            nn.BatchNorm2d(128) if args.discriminator_batchnorm else None,
            nn.LeakyReLU(),
            Reshape(-1, 128 * 7 * 7),  # N, 128*7*7
            nn.Linear(128 * 7 * 7, 1024),  # N, 1024
            nn.BatchNorm1d(1024) if args.discriminator_batchnorm else None,
            nn.LeakyReLU(),
            nn.Linear(1024, 1),  # N, 1
            Reshape(-1)] if m is not None])  # N


class GANModel(nn.Module):
    # GAN containing generator and discriminator
    def __init__(self, args, discriminator, generator):
        super(GANModel, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = args.latent_dim
        self._state_hooks = {}  # used by inferno for logging
        self.apply(initializer)  # initialize the parameters

    def generate(self, latent):
        # Generate fake images from latent inputs
        xfake = self.generator(latent)
        # Save images for later
        self._state_hooks['xfake'] = xfake
        self._state_hooks['generated_images'] = format_images(xfake)  # log the generated images
        return xfake

    def discriminate(self, x):
        # Run discriminator on an input
        return self.discriminator(x)

    def y_fake(self, latent):
        # Run discriminator on generated images
        yfake = self.discriminate(self.generate(latent))
        return yfake

    def y_real(self, xreal):
        # Run discriminator on real images
        yreal = self.discriminate(xreal)
        # Save images for later
        self._state_hooks['xreal'] = xreal
        self._state_hooks['real_images'] = format_images(xreal)
        return yreal

    def latent_sample(self, xreal):
        # Generate latent samples of same shape as real data
        latent = xreal.data.new(xreal.size(0), self.latent_dim)
        torch.randn(*latent.size(), out=latent)
        latent = Variable(latent)
        return latent

    def forward(self, xreal):
        # Calculate and return y_real and y_fake
        return self.y_real(xreal), self.y_fake(self.latent_sample(xreal))


class DiscriminatorLoss(nn.BCEWithLogitsLoss):
    # Loss function for discriminator
    def forward(self, input, _):
        # Targets are ignored because we know they are 0 or 1
        yreal, yfake = input  # unpack inputs
        real_targets = Variable(yreal.data.new(yreal.size(0)).fill_(1))  # targets for real images
        fake_targets = Variable(yreal.data.new(yreal.size(0)).zero_())  # targets for generated images
        real_loss = super(DiscriminatorLoss, self).forward(yreal, real_targets)  # loss for real images
        fake_loss = super(DiscriminatorLoss, self).forward(yfake, fake_targets)  # loss for fake images
        loss = real_loss + fake_loss  # combined loss
        return loss


class GeneratorLoss(nn.BCEWithLogitsLoss):
    # Loss function for generator
    def forward(self, yfake):
        # No targets because we know the targets
        fake_targets = Variable(yfake.data.new(yfake.size(0)).fill_(1))  # targets for fake images
        fake_loss = super(GeneratorLoss, self).forward(yfake, fake_targets)  # loss for fake images
        return fake_loss


class GeneratorTrainingCallback(Callback):
    # Callback periodically trains the generator
    def __init__(self, args, parameters, criterion):
        self.criterion = criterion
        self.opt = Adam(parameters, args.generator_lr)
        self.batch_size = args.batch_size
        self.latent_dim = args.latent_dim
        self.count = 0
        self.frequency = args.generator_frequency

    def end_of_training_iteration(self, **_):
        # Each iteration check if it is time to train the generator
        self.count += 1
        if self.count > self.frequency:
            self.train_generator()
            self.count = 0

    def train_generator(self):
        # Train the generator
        # Generate latent samples
        if self.trainer.is_cuda():
            latent = torch.cuda.FloatTensor(self.batch_size, self.latent_dim)
        else:
            latent = torch.FloatTensor(self.batch_size, self.latent_dim)
        torch.randn(*latent.size(), out=latent)
        latent = Variable(latent)
        # Calculate yfake
        yfake = self.trainer.model.y_fake(latent)
        # Calculate loss
        loss = self.criterion(yfake)
        # Perform update
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


class GenerateDataCallback(Callback):
    # Callback saves generated images to a folder
    def __init__(self, args, gridsize=10):
        super(GenerateDataCallback, self).__init__()
        self.count = 0  # iteration counter
        self.image_count = 0  # image counter
        self.frequency = args.image_frequency
        self.gridsize = gridsize
        self.latent = torch.randn(gridsize * gridsize, args.latent_dim)

    def end_of_training_iteration(self, **_):
        # Check if it is time to generate images
        self.count += 1
        if self.count > self.frequency:
            self.save_images()
            self.count = 0

    def generate(self, latent):
        # Set eval, generate, then set back to train
        self.trainer.model.eval()
        generated = self.trainer.model.generate(Variable(latent))
        self.trainer.model.train()
        return generated

    def save_images(self):
        # Generate images
        path = os.path.join(self.trainer.save_directory, 'generated_images')
        os.makedirs(path, exist_ok=True)  # create directory if necessary
        image_path = os.path.join(path, '{:08d}.png'.format(self.image_count))
        self.image_count += 1
        # Copy latent to cuda if necessary
        if self.trainer.is_cuda():
            latent = self.latent.cuda()
        else:
            latent = self.latent
        generated = self.generate(latent)
        # Reshape, scale, and cast the data so it can be saved
        grid = format_images(generated).squeeze(0).permute(1, 2, 0)
        if grid.size(2) == 1:
            grid = grid.squeeze(2)
        array = grid.data.cpu().numpy() * 255.
        array = array.astype(np.uint8)
        # Save the image
        Image.fromarray(array).save(image_path)


def run(args):
    save_args(args)  # save command line to a file for reference
    train_loader = mnist_data_loader(args)  # get the data
    # Create the model
    model = GANModel(
        args,
        discriminator=DiscriminatorNetwork(args),
        generator=GeneratorNetwork(args))

    # Build trainer
    trainer = Trainer(model)
    trainer.build_criterion(DiscriminatorLoss)
    trainer.build_optimizer('Adam', model.discriminator.parameters(), lr=args.discriminator_lr)
    trainer.save_every((1, 'epochs'))
    trainer.save_to_directory(args.save_directory)
    trainer.set_max_num_epochs(args.epochs)
    trainer.register_callback(GenerateDataCallback(args))
    trainer.register_callback(GeneratorTrainingCallback(
        args,
        parameters=model.generator.parameters(),
        criterion=GeneratorLoss()))
    trainer.bind_loader('train', train_loader)
    # Custom logging configuration so it knows to log our images
    logger = TensorboardLogger(
        log_scalars_every=(1, 'iteration'),
        log_images_every=(args.log_image_frequency, 'iteration'))
    trainer.build_logger(logger, log_directory=args.save_directory)
    logger.observe_state('generated_images')
    logger.observe_state('real_images')
    logger._trainer_states_being_observed_while_training.remove('training_inputs')

    if args.cuda:
        trainer.cuda()

    # Go!
    trainer.fit()

    # Generate video from saved images
    if not args.no_ffmpeg:
        generate_video(args.save_directory)


def main(argv):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch GAN Example')

    # Output directory
    parser.add_argument('--save-directory', type=str, default='output/mnist_gan/v1', help='output directory')

    # Configuration
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='batch size')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs')
    parser.add_argument('--image-frequency', type=int, default=10, metavar='N', help='frequency to write images')
    parser.add_argument('--log-image-frequency', type=int, default=100, metavar='N', help='frequency to log images')
    parser.add_argument('--generator-frequency', type=int, default=5, metavar='N', help='frequency to train generator')

    # Hyperparameters
    parser.add_argument('--latent-dim', type=int, default=100, metavar='N', help='latent dimension')
    parser.add_argument('--discriminator-lr', type=float, default=3e-4, metavar='N', help='discriminator learning rate')
    parser.add_argument('--generator-lr', type=float, default=3e-4, metavar='N', help='generator learning rate')
    parser.add_argument('--discriminator-batchnorm', type=bool, default=True, metavar='N', help='enable BN')
    parser.add_argument('--generator-batchnorm', type=bool, default=True, metavar='N', help='enable BN')

    # Flags
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--no-ffmpeg', action='store_true', default=False, help='disables video generation')

    args = parser.parse_args(argv)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
