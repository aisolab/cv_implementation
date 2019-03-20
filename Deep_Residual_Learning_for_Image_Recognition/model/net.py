import torch
import torch.nn as nn
import torch.nn.functional as F
# https://arxiv.org/abs/1512.03385

class Bottleneck(nn.Module):
    """Bottleneck building block with identity shortcut"""

    def __init__(self, in_channels: int) -> None:
        """Instantiating Bottleneck class

        Args:
            in_channels (int): the number of in_channels
        """
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
                                        nn.BatchNorm2d(in_channels // 4),
                                        nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(in_channels // 4),
                                        nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
                                        nn.BatchNorm2d(in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        featuremap = F.relu(self.bottleneck(x) + x)
        return featuremap


