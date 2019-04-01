import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    """Flatten class"""
    def forward(self, x):
        return x.view(x.size(0), -1)

class Bottleneck(nn.Module):
    """Bottleneck class"""
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self._pooling = (in_channels != out_channels)

        if self._pooling:
            self._ops = nn.Sequential(nn.Conv2d(in_channels, in_channels // 2, 1, 1),
                                      nn.BatchNorm2d(in_channels // 2),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels // 2, in_channels // 2, 3, 2, 1),
                                      nn.BatchNorm2d(in_channels // 2),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels // 2, out_channels, 1, 1),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU())
            self._shortcut = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        else:
            self._ops = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4, 1, 1),
                                      nn.BatchNorm2d(in_channels // 4),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels // 4, in_channels // 4, 3, 1, 1),
                                      nn.BatchNorm2d(in_channels // 4),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels // 4, out_channels, 1, 1),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU())
    def forward(self, x):
        shortcut = self._shortcut(x) if self._pooling else x
        fmap = F.relu(self._ops(x) + shortcut)
        return fmap

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self._ops = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1),
                                  nn.MaxPool2d(3, 2, 1),
                                  Bottleneck(64, 64),
                                  Bottleneck(64, 64),
                                  Bottleneck(64, 64),
                                  Bottleneck(64, 128),
                                  Bottleneck(128, 128),
                                  Bottleneck(128, 128),
                                  Bottleneck(128, 128),
                                  Bottleneck(128, 256),
                                  Bottleneck(256, 256),
                                  Bottleneck(256, 256),
                                  Bottleneck(256, 256),
                                  Bottleneck(256, 256),
                                  Bottleneck(256, 256),
                                  Bottleneck(256, 512),
                                  Bottleneck(512, 512),
                                  Bottleneck(512, 512),
                                  nn.AdaptiveAvgPool2d((1, 1)),
                                  nn.Conv2d(512, num_classes, 1, 1),
                                  Flatten())

    def forward(self, x):
        score = self._ops(x)
        return score
