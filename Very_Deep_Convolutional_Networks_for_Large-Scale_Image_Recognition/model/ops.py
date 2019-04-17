import torch
import torch.nn as nn


class Flatten(nn.Module):
    """Flatten class"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class ConvBlock(nn.Module):
    "ConvBlock for Vgg16"
    def __init__(self, in_channels: int, out_channels: int, use_1x1conv: bool) -> None:
        """Instantiating ConvBlock class

        Args:
            in_channels (int): the number of channels from input featuremap
            out_channels (int): the number of channels from output featuremap
            use_1x1conv (bool): Using 1x1 convolution
        """
        super(ConvBlock, self).__init__()
        self._ops = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(out_channels),
                                  nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(out_channels))

        if use_1x1conv == True:
            self._ops.add_module(nn.Conv2d.__name__, nn.Conv2d(out_channels, out_channels, 1, 1, 0))
            self._ops.add_module(nn.ReLU.__name__, nn.ReLU())
            self._ops.add_module(nn.BatchNorm2d.__name__, nn.BatchNorm2d(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self._ops(x)
        return fmap