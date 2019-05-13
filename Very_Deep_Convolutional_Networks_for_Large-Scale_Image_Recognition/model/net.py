from model.ops import Flatten, ConvBlock
import torch
import torch.nn as nn


class Vgg16(nn.Module):
    """Vgg16 in https://arxiv.org/abs/1409.1556"""
    def __init__(self, num_classes: int) -> None:
        """Instantiating Vgg16 class

        Args:
            num_classes (int): the number of classes
        """
        super(Vgg16, self).__init__()
        self._ops = nn.Sequential(ConvBlock(3, 64, False),
                                  nn.MaxPool2d(2, 2),
                                  ConvBlock(64, 128, False),
                                  nn.MaxPool2d(2, 2),
                                  ConvBlock(128, 256, True),
                                  nn.MaxPool2d(2, 2),
                                  ConvBlock(256, 512, True),
                                  nn.MaxPool2d(2, 2),
                                  ConvBlock(512, 512, True),
                                  nn.AdaptiveAvgPool2d((1, 1)),
                                  Flatten(),
                                  nn.Linear(512, 512),
                                  nn.BatchNorm1d(512),
                                  nn.ReLU(),
                                  nn.Linear(512, 512),
                                  nn.BatchNorm1d(512),
                                  nn.ReLU(),
                                  nn.Linear(512, num_classes))

        self.apply(self._init_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self._ops(x)
        return score

    def _init_weight(self, layer):
        nn.init.kaiming_uniform_(layer.weight) if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) \
            else None
