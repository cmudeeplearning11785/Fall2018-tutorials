import argparse
import sys

import torch
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from torch import nn
from torch.autograd import Variable
from torch.autograd import grad

from mnist_gan import generate_video, mnist_data_loader, DiscriminatorNetwork, GeneratorNetwork
from mnist_gan import save_args, GANModel, GenerateDataCallback, GeneratorTrainingCallback


class WGANDiscriminatorLoss(nn.Module):
    def __init__(self, penalty_weight, model):
        super(WGANDiscriminatorLoss, self).__init__()
        self.model = model
        self.penalty_weight = penalty_weight

    # Run discriminator
    def discriminate(self, xmix):
        return self.model.discriminate(xmix)

    # Loss function for discriminator
    def forward(self, input, _):
        # Targets are ignored
        yreal, yfake = input  # unpack inputs

        # Main loss calculation
        wgan_loss = yfake.mean() - yreal.mean()

        # Gradient penalty
        xreal = self.model._state_hooks['xreal']
        xfake = self.model._state_hooks['xfake']
        # Random linear combination of xreal and xfake
        alpha = Variable(torch.rand(xreal.size(0), 1, 1, 1, out=xreal.data.new()))
        xmix = (alpha * xreal) + ((1. - alpha) * xfake)
        # Run discriminator on the combination
        ymix = self.discriminate(xmix)
        # Calculate gradient of output w.r.t. input
        ysum = ymix.sum()
        grads = grad(ysum, [xmix], create_graph=True)[0]
        gradnorm = torch.sqrt((grads * grads).sum(3).sum(2).sum(1))
        graddiff = gradnorm - 1
        gradpenalty = (graddiff * graddiff).mean() * self.penalty_weight

        # Total loss
        loss = wgan_loss + gradpenalty
        return loss


class WGANGeneratorLoss(nn.BCEWithLogitsLoss):
    # Loss function for generator
    def forward(self, yfake):
        loss = -yfake.mean()
        return loss


def run(args):
    save_args(args)  # save command line to a file for reference
    train_loader = mnist_data_loader(args)  # get the data
    model = GANModel(
        args,
        discriminator=DiscriminatorNetwork(args),
        generator=GeneratorNetwork(args))

    # Build trainer
    trainer = Trainer(model)
    trainer.build_criterion(WGANDiscriminatorLoss(penalty_weight=args.penalty_weight, model=model))
    trainer.build_optimizer('Adam', model.discriminator.parameters(), lr=args.discriminator_lr)
    trainer.save_every((1, 'epochs'))
    trainer.save_to_directory(args.save_directory)
    trainer.set_max_num_epochs(args.epochs)
    trainer.register_callback(GenerateDataCallback(args))
    trainer.register_callback(GeneratorTrainingCallback(
        args,
        parameters=model.generator.parameters(),
        criterion=WGANGeneratorLoss()))
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
    parser.add_argument('--save-directory', type=str, default='output/mnist_wgangp/v1', help='output directory')

    # Configuration
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='batch size')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs')
    parser.add_argument('--image-frequency', type=int, default=10, metavar='N', help='frequency to write images')
    parser.add_argument('--log-image-frequency', type=int, default=100, metavar='N', help='frequency to log images')
    parser.add_argument('--generator-frequency', type=int, default=10, metavar='N', help='frequency to train generator')

    # Hyperparameters
    parser.add_argument('--latent-dim', type=int, default=100, metavar='N', help='latent dimension')
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
