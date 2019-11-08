import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import conv
from layers import lrelu, relu, tanh
from layers import batch_norm, spectral_norm
from layers import SelfAttn


class Discriminator(nn.Module):
    """SAGAN Discriminator"""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()

        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        output = []

        # layer 1

        # 3 -> 64
        layer1.append(spectral_norm(
            conv(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        layer1.append(lrelu())

        # layer 2
        input_dim = conv_dim
        output_dim = input_dim*2

        # 64 -> 128
        layer2.append(spectral_norm(
            conv(input_dim, output_dim, kernel_size=4, stride=2, padding=1)))
        layer2.append(lrelu())

        # layer 3
        input_dim = output_dim
        output_dim = input_dim*2

        # 128 -> 256
        layer3.append(spectral_norm(
            conv(input_dim, output_dim, kernel_size=4, stride=2, padding=1)))
        layer3.append(lrelu())

        # layer 4
        input_dim = output_dim
        output_dim = input_dim*2

        # 256 -> 512
        layer4.append(spectral_norm(
            conv(input_dim, output_dim, kernel_size=4, stride=2, padding=1)))
        layer4.append(lrelu())

        # output layer
        input_dim = input_dim*2

        # 512 -> 1
        output.append(conv(input_dim, 1, kernel_size=4))

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.attn1 = SelfAttn(256)
        self.l4 = nn.Sequential(*layer4)
        self.attn2 = SelfAttn(512)
        self.output = nn.Sequential(*output)

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out, b1 = self.attn1(out)
        out = self.l4(out)
        out, b2 = self.attn2(out)
        out = self.output(out)

        return out.squeeze(), b1, b2
