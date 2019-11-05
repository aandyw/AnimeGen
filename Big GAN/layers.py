import torch
import torch.nn as nn
import torch.nn.functional as F


# getting total parameters of network
def parameters(network):
    # sum params of each layer
    params = list(p.numel() for p in network.parameters())
    return sum(params)


# conv 3x3 layer for nn
def conv3x3(input_size, output_size):
    return nn.Conv2d(input_size, output_size,
                     kernel_size=3, stride=1, padding=0, dilation=1, bias=False)


# conv 1x1 layer for nn
# feature maps
def conv1x1(input_size, output_size):
    return nn.Conv2d(input_size, output_size,
                     kernel_size=1, stride=1, padding=0, dilation=1, bias=False)


def init_weights(m):
    pass


##################################################################################
# Self Attention
##################################################################################
class SelfAttn(nn.Module):
    def __init__(self, channels):
        super(self.__class__.__name__, self).__init__()

    def forward(self, t):
        pass

##################################################################################
# Conditional Batch Norm
##################################################################################
class ConditionalNorm(nn.Module):
    def __init__(self, input_size, n_condition):
        super(self.__class__.__name__, self).__init__()
        self.bn = nn.BatchNorm2d(input_size, affine=True)
        self.embed = nn.Linear(input_size, n_condition*2)

        nn.init.orthogonal_(self.embed.weight.data[:, :input_size], gain=1)
        self.embed.weight.data[:, input_size:].zero_()

    def forward(self, inputs, labels):
        out = self.bn(inputs)
        embed = self.embed(labels.float())
        gamma, beta = embed.chunk(2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        out = gamma * out + beta
        return out


# resblock up
class ResidualBlock_G(nn.Module):
    def __init__(self, input_size, output_size, label_dims):
        super(self.__class__.__name__, self).__init__()
        self.cbn_1 = ConditionalNorm(input_size, label_dims)  # init cbn
        self.upsample = nn.Upsample(
            size=None, scale_factor=2, mode="nearest", align_corners=None)
        self.conv3x3_1 = conv3x3()

        self.cbn_2 = ConditionalNorm()
        self.conv3x3_2 = conv3x3()

        self. conv1x1 = conv1x1()

    def forward(self, inputs, labels):
        x = F.leaky_relu(self.cbn_1(inputs, labels))  # pass to cbn
        x = self.upsample(x)
        x = self.conv3x3_1(x)

        x = F.leaky_relu(self.cbn_2(x, labels))
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
