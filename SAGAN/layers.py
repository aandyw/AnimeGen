import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(input_size, output_size):
    return nn.Conv2d(input_size, output_size,
                     kernel_size=1, stride=1, padding=0, dilation=1, bias=False)


# https://arxiv.org/pdf/1805.08318.pdf
class SelfAttn(nn.Module):
    """ Self attention layer """

    def __init__(self, channels):
        super(self.__class__.__name__, self).__init__()
        k = channels // 8
        # getting feature maps for f(x), g(x) and h(x)
        self.f = conv1x1(channels, k)
        self.g = conv1x1(channels, k)
        self.h = conv1x1(channels, channels)
        self.gamma = nn.Parameter(torch.zeros(1))  # y = γo + x

        # softmax for matmul of f(x) & g(x)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs:
                x: conv feature maps (b*c*w*h)

            outputs:
                out: self attention feature maps (o)
                attention: attention map (b*n*n)
        """
        b, c, width, height = x.size()
        N = width*height
        f = self.f(x).view(b, -1, N).permute(0, 2, 1)  # b*c*n -> b*n*c
        g = self.g(x).view(b, -1, N) # b*c*n
        h = self.h(x).view(b, -1, N) # b*c*n

        # f (b x n x c) tensor, g (b x c x n) tensor
        # out tensor (b x n x n)
        transpose = torch.bmn(f, g)  # matmul
        attention = self.softmax(transpose).permute(0, 2, 1)  # b*n*n

        out = torch.bmn(h, attention)
        out = out.view(b, c, width, height)

        out = self.gamma*out + x
        return out, attention
