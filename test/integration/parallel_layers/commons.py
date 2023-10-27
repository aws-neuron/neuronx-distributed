import random

import numpy
import torch


def print_separator(message):
    torch.distributed.barrier()
    filler_len = (78 - len(message)) // 2
    filler = "-" * filler_len
    string = "\n" + filler + " {} ".format(message) + filler
    if torch.distributed.get_rank() == 0:
        print(string, flush=True)
    torch.distributed.barrier()


class IdentityLayer(torch.nn.Module):
    def __init__(self, size, scale=1.0):
        super(IdentityLayer, self).__init__()
        self.weight = torch.nn.Parameter(scale * torch.randn(size))

    def forward(self):
        return self.weight


class IdentityLayer3D(torch.nn.Module):
    def __init__(self, m, n, k):
        super(IdentityLayer3D, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(m, n, k))
        torch.nn.init.normal_(self.weight)

    def forward(self):
        return self.weight


def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
