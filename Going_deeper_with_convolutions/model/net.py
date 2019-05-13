import torch
import torch.nn as nn
from model.ops import InceptionV1, Flatten


class Googlenet(nn.Module):
    """Googlenet class"""
    def __init__(self, num_classes: int) -> None:
        """Instantiating Googlenet class

        Args:
            num_classes (int): the number of classes
        """
        super(Googlenet, self).__init__()
        self._ops = nn.Sequential(nn.Conv2d(3, 64, 1, 1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 96, 3, 1, 1),
                                  nn.BatchNorm2d(96),
                                  nn.ReLU(),
                                  InceptionV1(96, 32, 48, 64, 8, 16, 16),
                                  InceptionV1(128, 64, 64, 96, 16, 48, 32),
                                  nn.MaxPool2d(3, 2, 1),
                                  InceptionV1(240, 96, 48, 104, 8, 24, 32),
                                  InceptionV1(256, 80, 56, 112, 12, 32, 32),
                                  InceptionV1(256, 64, 64, 128, 12, 32, 32),
                                  InceptionV1(256, 56, 72, 144, 16, 32, 32),
                                  InceptionV1(264, 128, 80, 160, 16, 64, 64),
                                  nn.MaxPool2d(3, 2, 1),
                                  InceptionV1(416, 128, 80, 160, 16, 64, 64),
                                  InceptionV1(416, 192, 96, 192, 24, 64, 64),
                                  nn.AdaptiveMaxPool2d((1, 1)),
                                  Flatten(),
                                  nn.Dropout(),
                                  nn.Linear(512, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self._ops(x)
        return fmap