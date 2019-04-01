import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    """Bottleneck class"""
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self._ops

    def forward(self, x):