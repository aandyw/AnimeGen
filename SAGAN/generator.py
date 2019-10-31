import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import SelfAttn

class Generator(nn.Module):
    
    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64):
        super(self.__class__.__name__, self).__init__()
        
        self.attn = SelfAttn()
    def forward(self, z):
        return