import torch
import torch.nn as nn
import torch.nn.functional as F


# getting total parameters of network
def parameters(network):
    # sum params of each layer
    params = list(p.numel() for p in network.parameters())
    return sum(params)


# conv 3x3 layer for nn
def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=1, padding=0, dilation=1, bias=False)


# conv 1x1 layer for nn
def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=1, stride=1, padding=0, dilation=1, bias=False)


def init_weights(m):
    pass


class Self_Attn(nn.Module):
    def __init__(self):
        super(self.__class__.__name__, self).__init__()

    def forward(self, t):
        pass


class ConditionalBatchNorm(nn.Module):
    def __init__(self, in_channels):
        super(self.__class__.__name__, self).__init__()

    def forward(self, inputs, labels):
        pass


# resblock up
class ResidualBlock_G(nn.Module):
    def __init__(self, in_channels, out_channels, label_dims):
        super(self.__class__.__name__, self).__init__()
        self.batchnorm_1 = ConditionalBatchNorm()  # init cbn
        self.upsample = nn.Upsample(
            size=None, scale_factor=2, mode="nearest", align_corners=None)
        self.conv3x3_1 = conv3x3()

        self.batchnorm_2 = ConditionalBatchNorm()
        self.conv3x3_2 = conv3x3()

        self. conv1x1 = conv1x1()

    def forward(self, inputs, labels):
        x = F.leaky_relu(self.batchnorm_1(inputs, labels))  # pass to cbn
        x = self.upsample(x)
        x = self.conv3x3_1(x)

        x = F.leaky_relu(self.batchnorm_2(x, labels))
        x = self.conv3x3_2(x)

        x += self.conv3x3_1(self.upsample(inputs))
        return x


# resblock down
class ResidualBlock_D(nn.Module):
    def __init__(self):
        super(self.__class__.__name__, self).__init__()
        self.conv3x3_1 = conv3x3()
        self.conv3x3_2 = conv3x3()

    def forward(self, inputs):
        pass
