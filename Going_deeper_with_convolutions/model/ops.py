import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    """Flatten class"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class InceptionV1(nn.Module):
    """InceptionV1 class"""
    def __init__(self, in_channels, o_1: int, r_3: int, o_3: int, r_5: int, o_5: int, pool: int) -> None:
        """Instantiating InceptionV1 class

        Args:
            in_channels (int): the number of channels from input tensor
            o_1 (int): the number of output channels from 1x1 convolution
            r_3 (int): the number of output channels from 1x1 convolution in reduced 3x3 convolution
            o_3 (int): the number of output channels from 3x3 convolution in reduced 3x3 convolution
            r_5 (int): the number of output channels from 1x1 convolution in reduced 5x5 convolution
            o_5 (int): the number of output channels from 5x5 convolution in reduced 5x5 convolution
            pool (int): the number of output channels from 1x1 convolution following max pooling
        """
        super(InceptionV1, self).__init__()
        self._ops_1x1 = nn.Sequential(nn.Conv2d(in_channels, o_1, 1),
                                      nn.BatchNorm2d(o_1),
                                      nn.ReLU())
        self._ops_reduced_3x3 = nn.Sequential(nn.Conv2d(in_channels, r_3, 1),
                                              nn.BatchNorm2d(r_3),
                                              nn.ReLU(),
                                              nn.Conv2d(r_3, o_3, 3, padding=1),
                                              nn.BatchNorm2d(o_3),
                                              nn.ReLU())
        self._ops_reduced_5x5 = nn.Sequential(nn.Conv2d(in_channels, r_5, 1),
                                              nn.BatchNorm2d(r_5),
                                              nn.ReLU(),
                                              nn.Conv2d(r_5, o_5, 5, padding=2),
                                              nn.BatchNorm2d(o_5),
                                              nn.ReLU(),
                                              nn.BatchNorm2d(o_5))
        self._ops_maxpool_1x1 = nn.Sequential(nn.MaxPool2d(3, 1, padding=1),
                                              nn.Conv2d(in_channels, pool, 1),
                                              nn.BatchNorm2d(pool),
                                              nn.ReLU())
        self._bn = nn.BatchNorm2d(o_1 + o_3 + o_5 + pool)

    def forward(self, x) -> torch.Tensor:
        fmap_1x1 = self._ops_1x1(x)
        fmap_reduced_3x3 = self._ops_reduced_3x3(x)
        fmap_reduced_5x5 = self._ops_reduced_5x5(x)
        fmap_maxpool_1x1 = self._ops_maxpool_1x1(x)
        fmap = torch.cat([fmap_1x1, fmap_reduced_3x3, fmap_reduced_5x5, fmap_maxpool_1x1], dim=1)
        fmap = F.relu(self._bn(fmap))

        return fmap