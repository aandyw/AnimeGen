import torch
import torch.nn as nn
import torch.nn.functional as F


##################################################################################
# Layers
##################################################################################

# conv 3x3 layer for nn
def conv3x3(input_size, output_size):
    return nn.Conv2d(input_size, output_size,
                     kernel_size=3, stride=1, padding=1, bias=True)


# conv 1x1 layer for nn
# feature maps
def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=1, stride=1, padding=0, bias=True)


# # discriminator
# def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
#     return nn.Conv2d(in_channels, out_channels,
#                      kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

# # generator
# def deconv(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
#     return nn.ConvTranspose2d(in_channels, out_channels,
#                               kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


def linear(in_features, out_features):
    return nn.Linear(in_features, out_features, bias=True)


def embedding(num_embeddings, embedding_dim):
    return nn.Embedding(num_embeddings, embedding_dim)


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear or type(m) == nn.Embedding:
        nn.init.orthogonal_(m.weight, gain=1)


##################################################################################
# Activation Functions
##################################################################################
def lrelu(negative_slope=0.1, inplace=False):
    return nn.LeakyReLU(negative_slope, inplace)


def relu(inplace=False):
    return nn.ReLU(inplace)


def tanh():
    return nn.Tanh()


##################################################################################
# Normalization Functions
##################################################################################
def batch_norm(num_features, eps=1e-05, momentum=0.1, affine=True):
    return nn.BatchNorm2d(num_features, eps, momentum, affine)


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
        self.f = spectral_norm(conv1x1(channels, k))  # [b, k, w, h]
        self.g = spectral_norm(conv1x1(channels, k))  # [b, k, w, h]
        self.h = spectral_norm(
            conv1x1(channels, channels // 2))  # [b, c//2, w, h]
        self.v = spectral_norm(
            conv1x1(channels // 2, channels))  # [b, c//2, w, h]
        self.gamma = nn.Parameter(torch.zeros(1))  # y = γo + x

        # softmax for matmul of f(x) & g(x)
        self.softmax = nn.Softmax(dim=-1)

        self.apply(init_weights)

    def forward(self, x):
        """
            inputs:
                x: conv feature maps (b*c*w*h)

            outputs:
                out: self attention feature maps (o)
                attention: attention map (b*n*n)
        """
        b, c, width, height = x.size()
        N = width * height
        f = self.f(x).view(b, -1, N).permute(0, 2, 1)  # b*c*n -> b*n*c
        g = self.g(x).view(b, -1, N)  # b*c*n
        h = self.h(x).view(b, -1, N)  # b*c*n

        # f (b x n x c) tensor, g (b x c x n) tensor
        # out tensor (b x n x n)
        transpose = torch.bmm(f, g)  # matmul
        beta = self.softmax(transpose).permute(0, 2, 1)  # b*n*n

        out = torch.bmm(h, beta)
        out = self.v(out.view(b, c // 2, width, height))

        out = self.gamma * out + x
        return out


##################################################################################
# Conditional Batch Norm
# https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
##################################################################################
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = batch_norm(num_features, momentum=0.001, affine=False)
        self.embed = embedding(num_classes, num_features * 2)
        # self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, :num_features].fill_(1.)  # init scale to 1
        self.embed.weight.data[:, num_features:].zero_()  # init bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + \
            beta.view(-1, self.num_features, 1, 1)
        return out


##################################################################################
# Residual Blocks
# https://arxiv.org/pdf/1809.11096.pdf (Architectural Details)
##################################################################################

# resblock up
class ResidualBlock_G(nn.Module):
    def __init__(self, in_channels, out_channels, label_dims):
        super(ResidualBlock_G, self).__init__()
        self.cbn_1 = ConditionalBatchNorm2d(in_channels, label_dims)
        self.lrelu = lrelu(inplace=True)
        self.upsample = nn.Upsample(
            size=None, scale_factor=2, mode="nearest", align_corners=None)
        self.conv3x3_1 = spectral_norm(conv3x3(in_channels, out_channels))

        self.cbn_2 = ConditionalBatchNorm2d(out_channels, label_dims)
        self.conv3x3_2 = spectral_norm(conv3x3(out_channels, out_channels))

        self.conv1x1 = spectral_norm(conv1x1(in_channels, out_channels))

        self.apply(init_weights)

    def forward(self, inputs, labels):
        x = self.cbn_1(inputs, labels)
        x = self.lrelu(x)
        x = self.upsample(x)
        x = self.conv3x3_1(x)

        x = self.cbn_2(x, labels)
        x = self.lrelu(x)
        x = self.conv3x3_2(x)

        # concatenate with direct upsampled feature maps
        x += self.conv1x1(self.upsample(inputs))
        return x


# resblock down
class ResidualBlock_D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock_D, self).__init__()
        self.lrelu = relu(negative_slope=0.2, inplace=True)
        self.conv3x3_1 = spectral_norm(conv3x3(in_channels, out_channels))
        self.conv3x3_2 = spectral_norm(conv3x3(out_channels, out_channels))
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv1x1 = spectral_norm((conv1x1(in_channels, out_channels)))

        self.apply(init_weights)

    def forward(self, inputs):
        x = self.lrelu(inputs)
        x = self.conv3x3_1(x)
        x = self.lrelu(x)
        x = self.conv3x3_2(x)
        x = self.downsample(x)

        # concatenate downsampled feature maps
        x += self.downsample(self.conv1x1(inputs))
        return x


##################################################################################
# Loss Functions
##################################################################################
def loss_hinge_dis_real(d_real):
    """Hinge loss for discriminator with real outputs"""
    d_loss_real = torch.mean(F.relu(1.0 - d_real))
    return d_loss_real


def loss_hinge_dis_fake(d_fake):
    """Hinge loss for discriminator with fake outputs"""
    d_loss_fake = torch.mean(F.relu(1.0 + d_fake))
    return d_loss_fake


def loss_hinge_gen(d_fake):
    """Hinge loss for generator"""
    g_loss = -torch.mean(d_fake)
    return g_loss


##################################################################################
# Utilities
##################################################################################
# getting total parameters of network
def parameters(network):
    # sum params of each layer
    params = list(p.numel() for p in network.parameters())
    return sum(params)


def tensor2var(x, grad=False):
    """Tensor to Variable"""
    if torch.cuda.is_available():
        x = x.cuda()
    return torch.autograd.Variable(x, requires_grad=grad)


def denorm(x):
    """Denormalize Images"""
    out = (x + 1) / 2
    return out.clamp_(0, 1)
