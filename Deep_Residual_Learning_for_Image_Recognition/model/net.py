import torch.nn as nn
from model.ops import Flatten, BottleNeck


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self._ops = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
                                  nn.BatchNorm2d(64),
                                  BottleNeck(64, 64),
                                  BottleNeck(64, 64),
                                  BottleNeck(64, 128),
                                  BottleNeck(128, 128),
                                  BottleNeck(128, 128),
                                  BottleNeck(128, 128),
                                  BottleNeck(128, 256),
                                  BottleNeck(256, 256),
                                  BottleNeck(256, 256),
                                  BottleNeck(256, 256),
                                  BottleNeck(256, 256),
                                  BottleNeck(256, 256),
                                  BottleNeck(256, 512),
                                  BottleNeck(512, 512),
                                  BottleNeck(512, 512),
                                  BottleNeck(512, 512),
                                  nn.AdaptiveAvgPool2d((1, 1)),
                                  Flatten(),
                                  nn.Linear(512, num_classes))

    def forward(self, x):
        score = self._ops(x)
        return score

