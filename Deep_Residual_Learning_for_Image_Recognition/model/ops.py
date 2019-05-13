import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    """Flatten class"""
    def forward(self, x):
        return x.view(x.size(0), -1)


class BottleNeck(nn.Module):
    """Bottleneck class"""
    def __init__(self, in_channels, out_channels):
        super(BottleNeck, self).__init__()
        self._pooling = (in_channels != out_channels)

        if self._pooling:
            self._ops = nn.Sequential(nn.Conv2d(in_channels, in_channels // 2, 1, 1),
                                      nn.BatchNorm2d(in_channels // 2),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels // 2, in_channels // 2, 3, 2, 1),
                                      nn.BatchNorm2d(in_channels // 2),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels // 2, out_channels, 1, 1),
                                      nn.BatchNorm2d(out_channels))
            self._shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, 2),
                                           nn.BatchNorm2d(out_channels))
        else:
            self._ops = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4, 1, 1),
                                      nn.BatchNorm2d(in_channels // 4),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels // 4, in_channels // 4, 3, 1, 1),
                                      nn.BatchNorm2d(in_channels // 4),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels // 4, out_channels, 1, 1),
                                      nn.BatchNorm2d(out_channels))

        self._bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        shortcut = self._shortcut(x) if self._pooling else x
        fmap = self._ops(x) + shortcut
        fmap = F.relu(self._bn(fmap))
        return fmap
