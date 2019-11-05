from layers import ResidualBlock_G, SelfAttn
from layers import conv3x3, conv1x1


class Generator(nn.Module):
    def __init__(self):
        super(self.__class__.__name__, self).__init__()

        self.attn1 = SelfAttn()

    def forward(self, t):
        pass
