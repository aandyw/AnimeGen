import torch
import torch.nn as nn
import torch.nn.functional as F


##################################################################################
# Layers
##################################################################################
def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=1, stride=1, padding=0, dilation=1, bias=True)


# discriminator
def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


# generator
def deconv(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    return nn.ConvTranspose2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


##################################################################################
# Activation Functions
##################################################################################
def lrelu(negative_slope=0.1):
    return nn.LeakyReLU(negative_slope)


def relu():
    return nn.ReLU()


def tanh():
    return nn.Tanh()


##################################################################################
# Normalization Functions
##################################################################################
def batch_norm(num_features, eps=1e-05, momentum=0.1):
    return nn.BatchNorm2d(num_features, eps, momentum)


def spectral_norm(module):
    return nn.utils.spectral_norm(module)


##################################################################################
# Self Attention
# https://arxiv.org/pdf/1805.08318.pdf
##################################################################################
class SelfAttn(nn.Module):
    """ Self attention layer """

    def __init__(self, channels):
        super(SelfAttn, self).__init__()
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
        f = self.f(x).view(b, -1, N)
        f = f.permute(0, 2, 1)  # b*c*n -> b*n*c
        g = self.g(x).view(b, -1, N)  # b*c*n
        h = self.h(x).view(b, -1, N)  # b*c*n

        # f (b x n x c) tensor, g (b x c x n) tensor
        # out tensor (b x n x n)
        transpose = torch.bmm(f, g)  # matmul
        beta = self.softmax(transpose).permute(0, 2, 1)  # b*n*n

        out = torch.bmm(h, beta)
        out = out.view(b, c, width, height)

        out = self.gamma*out + x
        return out, beta

##################################################################################
# Utilities
##################################################################################
def tensor2var(x, grad=False):
    """Tensor to Variable"""
    if torch.cuda.is_available():
        x = x.cuda()
    return torch.autograd.Variable(x, requires_grad=grad)

def var2tensor(x):
    """Variable to Tensor"""
    return x.data.cpu()

def var2numpy(x):
    return x.data.cpu().numpy()

def denorm(x):
    """Denormalize Images"""
    out = (x + 1) / 2
    return out.clamp_(0, 1)
