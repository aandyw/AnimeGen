import torch
import torch.nn.functional as F


# Hinge loss
def loss_hinge_dis(d_real, d_fake):
    d_loss_real = torch.mean(F.relu(1.0 - d_real))
    d_loss_fake = torch.mean(F.relu(1.0 + d_fake))
    return d_loss_real, d_loss_fake


def loss_hinge_gen(d_fake):
    g_loss_fake = -torch.mean(d_fake)
    return g_loss_fake
