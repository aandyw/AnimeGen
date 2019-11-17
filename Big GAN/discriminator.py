import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import ResidualBlock_D, SelfAttn
from layers import init_weights, linear, embedding
from layers import spectral_norm
from layers import lrelu, relu


class Discriminator(nn.Module):
    """Big GAN Discriminator"""

    def __init__(self, conv_dim, num_classes):
        super(Discriminator, self).__init__()
        self.conv_dim = conv_dim

        self.res_1 = ResidualBlock_D(3, conv_dim)
        self.res_2 = ResidualBlock_D(conv_dim, conv_dim*2)
        self.attn = SelfAttn(conv_dim*2)
        self.res_3 = ResidualBlock_D(conv_dim*2, conv_dim*4)
        self.res_4 = ResidualBlock_D(conv_dim*4, conv_dim*8)
        self.res_5 = ResidualBlock_D(conv_dim*8, conv_dim*16)
        self.lrelu = lrelu(inplace=True)
        self.linear = spectral_norm(linear(conv_dim*16, 1))
        self.embed = embedding(num_classes, conv_dim*16)

        self.apply(init_weights)

    def forward(self, x, labels):
        out = self.res_1(x)  # [n, 3, 128, 128]->[n, conv_dim, 64, 64]
        out = self.res_2(out)  # [n, conv_dim*2, 32, 32]
        out = self.atnn(out)  # [n, conv_dim*2, 32, 32]
        out = self.res_3(out)  # [n, conv_dim*4, 16, 16]
        out = self.res_4(out)  # [n, conv_dim*8, 8, 8]
        out = self.res_5(out)  # [n, conv_dim*16, 4, 4]
        out = self.lrelu(out)  # [n, conv_dim*16, 4, 4]
        out = torch.sum(out, dim=[2, 3])  # [n, conv_dim*16]
        out = torch.squeeze(self.linear(out))

        # discriminator projection
        h_labels = self.embed(labels)  # [n, conv_dim*16]
        proj = torch.mul(out, h_labels)  # [n, conv_dim*16]
        out += torch.sum(proj, dim=[1])  # n
        return out
