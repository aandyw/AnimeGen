import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import ResidualBlock_G, SelfAttn
from layers import init_weights, linear, conv3x3
from layers import spectral_norm, batch_norm
from layers import relu, lrelu, tanh


class Generator(nn.Module):
    """Big GAN Generator"""

    def __init__(self, z_dim, conv_dim, num_classes):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim

        self.linear = spectral_norm(
            linear(in_features=z_dim, out_features=conv_dim*16*4*4))
        self.res_1 = ResidualBlock_G(conv_dim*16, conv_dim*16, num_classes)
        self.res_2 = ResidualBlock_G(conv_dim*16, conv_dim*8, num_classes)
        self.res_3 = ResidualBlock_G(conv_dim*8, conv_dim*4, num_classes)
        self.attn = SelfAttn(conv_dim*4)
        self.res_4 = ResidualBlock_G(conv_dim*4, conv_dim*2, num_classes)
        self.res_5 = ResidualBlock_G(conv_dim*2, conv_dim, num_classes)
        self.bn = batch_norm(conv_dim, eps=1e-5, momentum=0.0001)
        self.lrelu = lrelu(inplace=True)
        self.conv3x3 = spectral_norm(conv3x3(conv_dim, 3))
        self.tanh = tanh()

        self.apply(init_weights)

    def forward(self, z, labels):
        out = self.linear(z)  # [n, z_dim]->[n, conv_dim*16*4*4]
        out = out.view(-1, self.conv_dim*16, 4, 4)  # [n, conv_dim*16, 4, 4]
        out = self.res_1(out, labels)  # [n, conv_dim*16, 8, 8]
        out = self.res_2(out, labels)  # [n, conv_dim*8, 16, 16]
        out = self.res_3(out, labels)  # [n, conv_dim*4, 32, 32]
        out = self.attn(out)  # [n, conv_dim*4, 32, 32]
        out = self.res_4(out, labels)  # [n, conv_dim*2, 64, 64]
        out = self.res_5(out, labels)  # [n, conv_dim, 128, 128]
        out = self.bn(out)  # [n, conv_dim, 128, 128]
        out = self.lrelu(out)  # [n, conv_dim, 128, 128]
        out = self.conv3x3(out)  # [n, 3, 128, 128]
        out = self.tanh(out)  # [n, 3, 128, 128]
        return out
