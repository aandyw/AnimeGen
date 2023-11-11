import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import deconv
from layers import lrelu, relu, tanh
from layers import batch_norm, spectral_norm
from layers import SelfAttn


class Generator(nn.Module):
    """SAGAN Generator"""

    def __init__(self, image_size=64, z_dim=100, conv_dim=64):
        super(Generator, self).__init__()

        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        output = []

        # layer 1
        layer_num = int(np.log2(image_size)) - 3  # 3
        mult = 2 ** layer_num  # 8
        output_dim = conv_dim*mult  # 512

        # 100 -> 512
        layer1.append(spectral_norm(deconv(z_dim, output_dim, kernel_size=4)))
        layer1.append(batch_norm(output_dim))
        layer1.append(lrelu())

        # layer 2
        input_dim = output_dim
        output_dim = int(input_dim / 2)

        # 512 -> 256
        layer2.append(spectral_norm(
            deconv(input_dim, output_dim, kernel_size=4, stride=2, padding=1)))
        layer2.append(batch_norm(output_dim))
        layer2.append(lrelu())

        # layer 3
        input_dim = output_dim
        output_dim = int(input_dim / 2)

        # 256 -> 128
        layer3.append(spectral_norm(
            deconv(input_dim, output_dim, kernel_size=4, stride=2, padding=1)))
        layer3.append(batch_norm(output_dim))
        layer3.append(lrelu())

        # layer 4
        input_dim = output_dim
        output_dim = int(input_dim / 2)

        # 128 -> 64
        layer4.append(spectral_norm(
            deconv(input_dim, output_dim, kernel_size=4, stride=2, padding=1)))
        layer4.append(batch_norm(output_dim))
        layer4.append(lrelu())

        # output layer
        input_dim = output_dim

        # 64 -> 3
        output.append(deconv(input_dim, out_channels=3,
                             kernel_size=4, stride=2, padding=1))
        output.append(tanh())

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.attn1 = SelfAttn(128)
        self.l4 = nn.Sequential(*layer4)
        self.attn2 = SelfAttn(64)
        self.output = nn.Sequential(*output)

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)  # [b,100]->[b,100,1,1]
        out = self.l1(z)  # [b,100,1,1]->[b,512,4,4]
        out = self.l2(out)  # [b,512,4,4]->[b,256,8,8]
        out = self.l3(out)  # [b,256,8,8]->[b,128,16,16]
        out = self.attn1(out)
        out = self.l4(out)  # [b,128,16,16]->[b,64,32,32]
        out = self.attn2(out)
        out = self.output(out)  # [b,64,32,32]->[b,3,64,64]

        return out
