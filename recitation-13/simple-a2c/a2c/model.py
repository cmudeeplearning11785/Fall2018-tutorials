import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class SimpleResBlock(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super(SimpleResBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        z = F.elu(self.fc1(x))
        out = F.elu(self.fc2(z) + x)
        return out

class SimpleResNet(nn.Module):

    def __init__(self, input_size, output_size, hidden, num_layers):
        super(SimpleResNet, self).__init__()
        self.input_transform = nn.Linear(input_size, hidden)
        self.hidden_layers = nn.Sequential(*[SimpleResBlock(hidden, hidden, hidden) for _ in range(num_layers)])
        self.output_transform = nn.Linear(hidden, output_size)

    def forward(self, x):
        z1 = F.elu(self.input_transform(x))
        z2 = self.hidden_layers(z1)
        out = self.output_transform(z2)
        return out


class DiagonalGaussian(nn.Module):

    def __init__(self, input_size, output_size, fixed_std=None):
        super(DiagonalGaussian, self).__init__()
        self.mean_projection = nn.Linear(input_size, output_size)

        init_std = torch.ones(1, output_size)
        if fixed_std is None:
            self.logstd = nn.Parameter(init_std.log())
        else:
            self.logstd = init_std.fill_(fixed_std).log()

    def forward(self, x):
        dist = Normal(self.mean_projection(x), self.logstd.exp())
        return dist


class Actor(nn.Module):

    def __init__(self, input_dim, action_dim):
        super(Actor, self).__init__()
        hidden_size = 64
        num_layers = 3
        self.network = SimpleResNet(input_dim, hidden_size, hidden_size, num_layers)
        self.policy_dist = DiagonalGaussian(hidden_size, action_dim)


    def forward(self, x):
        z = self.network(x)
        return self.policy_dist(z)


class Critic(nn.Module):

    def __init__(self, input_dim):
        super(Critic, self).__init__()
        hidden_size = 128
        num_layers = 4
        self.network = SimpleResNet(input_dim, 1, hidden_size, num_layers)

    def forward(self, x):
        return self.network(x)
