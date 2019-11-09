import torch


def parameters(network):
    """Parameters in model"""
    params = list(p.numel() for p in network.parameters())
    return sum(params)


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
