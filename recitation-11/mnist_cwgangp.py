import argparse
import sys

import torch
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torchvision import transforms

from mnist_gan import Reshape, format_images
from mnist_gan import generate_video
from mnist_gan import save_args, GANModel, GenerateDataCallback, GeneratorTrainingCallback
from mnist_wgangp import WGANDiscriminatorLoss, WGANGeneratorLoss


class MNISTWrapper(Dataset):
    def __init__(self):
        super(MNISTWrapper, self).__init__()
        self.dataset = datasets.MNIST('./data/mnist', train=True, download=True,
                                      transform=transforms.ToTensor())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, y = self.dataset[item]
        return x, y, y


def mnist_cgan_data_loader(args):
    # Create DataLoader for MNIST
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    train_loader = DataLoader(
        MNISTWrapper(),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_loader


class CGeneratorNetwork(nn.Module):
    # Network for generation
    # Input is (N, latent_dim)
    def __init__(self, args):
        super(CGeneratorNetwork, self).__init__()
        self.embedding = nn.Embedding(10, args.embedding_dim)
        self.trunk = nn.Sequential(*[m for m in [
            nn.Linear(args.latent_dim + args.embedding_dim, 1024),
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

    def forward(self, latent, y):
        embedded = self.embedding(y)
        h = torch.cat((latent, embedded), dim=1)
        h = self.trunk(h)
        return h


class CDiscriminatorNetwork(nn.Module):
    # Network for discrimination
    # Input is (N, 1, 28, 28)
    def __init__(self, args):
        super(CDiscriminatorNetwork, self).__init__()
        self.embedding = nn.Embedding(10, args.embedding_dim)
        self.trunk = nn.Sequential(*[m for m in [
            nn.Conv2d(1 + args.embedding_dim, 64, kernel_size=4, stride=2, padding=1),  # N, 64, 14, 14
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

    def forward(self, x, y):
        embedded = self.embedding(y)  # (N, dim)
        embedded = embedded.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))
        h = torch.cat((x, embedded), dim=1)
        h = self.trunk(h)
        return h


class CGANModel(GANModel):
    # GAN containing generator and discriminator
    def __init__(self, args, discriminator, generator):
        super(CGANModel, self).__init__(
            args=args,
            discriminator=discriminator,
            generator=generator)

    def generate(self, latent, y):
        # Generate fake images from latent inputs
        xfake = self.generator(latent, y)
        # Save images for later
        self._state_hooks['xfake'] = xfake
        self._state_hooks['y'] = y
        self._state_hooks['generated_images'] = format_images(xfake)  # log the generated images
        return xfake

    def discriminate(self, x, y):
        # Run discriminator on an input
        return self.discriminator(x, y)

    def y_fake(self, latent, y):
        # Run discriminator on generated images
        yfake = self.discriminate(self.generate(latent, y), y)
        return yfake

    def y_real(self, xreal, y):
        # Run discriminator on real images
        yreal = self.discriminate(xreal, y)
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

    def forward(self, xreal, y):
        # Calculate and return y_real and y_fake
        return self.y_real(xreal, y), self.y_fake(self.latent_sample(xreal), y)


class CWGANDiscriminatorLoss(WGANDiscriminatorLoss):
    def discriminate(self, xmix):
        y = self.model._state_hooks['y']
        return self.model.discriminate(xmix, y)


class CGenerateDataCallback(GenerateDataCallback):
    # Callback saves generated images to a folder
    def __init__(self, args):
        super(CGenerateDataCallback, self).__init__(args, gridsize=10)
        self.y = torch.arange(0, 10).unsqueeze(1).expand(-1, 10).contiguous().view(-1).contiguous().long()

    def end_of_training_iteration(self, **_):
        # Check if it is time to generate images
        self.count += 1
        if self.count > self.frequency:
            self.save_images()
            self.count = 0

    def generate(self, latent):
        # Set eval, generate, then set back to train
        self.trainer.model.eval()
        y = Variable(self.y)
        if self.trainer.is_cuda():
            y = y.cuda()
        generated = self.trainer.model.generate(Variable(latent), y)
        self.trainer.model.train()
        return generated


class CGeneratorTrainingCallback(GeneratorTrainingCallback):
    # Callback periodically trains the generator
    def __init__(self, args, parameters, criterion):
        super(CGeneratorTrainingCallback, self).__init__(args, parameters, criterion)

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
        y = Variable(torch.rand(latent.size(0), out=latent.data.new()) * 10).long()
        yfake = self.trainer.model.y_fake(latent, y)
        # Calculate loss
        loss = self.criterion(yfake)
        # Perform update
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


def run(args):
    save_args(args)  # save command line to a file for reference
    train_loader = mnist_cgan_data_loader(args)  # get the data
    model = CGANModel(
        args,
        discriminator=CDiscriminatorNetwork(args),
        generator=CGeneratorNetwork(args))

    # Build trainer
    trainer = Trainer(model)
    trainer.build_criterion(CWGANDiscriminatorLoss(penalty_weight=args.penalty_weight, model=model))
    trainer.build_optimizer('Adam', model.discriminator.parameters(), lr=args.discriminator_lr)
    trainer.save_every((1, 'epochs'))
    trainer.save_to_directory(args.save_directory)
    trainer.set_max_num_epochs(args.epochs)
    trainer.register_callback(CGenerateDataCallback(args))
    trainer.register_callback(CGeneratorTrainingCallback(
        args,
        parameters=model.generator.parameters(),
        criterion=WGANGeneratorLoss()))
    trainer.bind_loader('train', train_loader, num_inputs=2)
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
    parser.add_argument('--save-directory', type=str, default='output/mnist_cwgangp/v1', help='output directory')

    # Configuration
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='batch size')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs')
    parser.add_argument('--image-frequency', type=int, default=10, metavar='N', help='frequency to write images')
    parser.add_argument('--log-image-frequency', type=int, default=100, metavar='N', help='frequency to log images')
    parser.add_argument('--generator-frequency', type=int, default=10, metavar='N', help='frequency to train generator')

    # Hyperparameters
    parser.add_argument('--latent-dim', type=int, default=100, metavar='N', help='latent dimension')
    parser.add_argument('--embedding-dim', type=int, default=32, metavar='N', help='latent dimension')
    parser.add_argument('--discriminator-lr', type=float, default=3e-4, metavar='N', help='discriminator learning rate')
    parser.add_argument('--generator-lr', type=float, default=3e-4, metavar='N', help='generator learning rate')
    parser.add_argument('--penalty-weight', type=float, default=20., metavar='N', help='gradient penalty weight')
    parser.add_argument('--discriminator-batchnorm', type=bool, default=False, metavar='N', help='enable BN')
    parser.add_argument('--generator-batchnorm', type=bool, default=True, metavar='N', help='enable BN')

    # Flags
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--no-ffmpeg', action='store_true', default=False, help='disables video generation')

    args = parser.parse_args(argv)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
