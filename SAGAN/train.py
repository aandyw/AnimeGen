import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import SAGAN.layers as layers
from SAGAN.generator import Generator
from SAGAN.discriminator import Discriminator


class SAGAN():
    def __init__(self, data_loader, configs):

        # Data Loader
        self.data_loader = data_loader
