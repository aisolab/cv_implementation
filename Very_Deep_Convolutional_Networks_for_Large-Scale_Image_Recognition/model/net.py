import torch
import torch.nn as nn
# https://arxiv.org/abs/1409.1556

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


class Vgg16(nn.Module):
    """Vgg16 in https://arxiv.org/abs/1409.1556"""
    def __init__(self, num_classes: int) -> None:
        """Instantiating Vgg16 class

        Args:
            num_classes (int): the number of classes
        """
        super(Vgg16, self).__init__()
        self._extractor = nn.Sequential(ConvBlock(3, 64, False),
                                        nn.MaxPool2d(2, 2),
                                        ConvBlock(64, 128, False),
                                        nn.MaxPool2d(2, 2),
                                        ConvBlock(128, 256, True),
                                        nn.MaxPool2d(2, 2),
                                        ConvBlock(256, 512, True),
                                        nn.MaxPool2d(2, 2),
                                        ConvBlock(512, 512, True),
                                        nn.MaxPool2d(2, 2))

        self._classifier = nn.Sequential(nn.Linear(512, 512),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(512),
                                         nn.Linear(512, 512),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(512),
                                         nn.Linear(512, num_classes))

        self.apply(self._init_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self._extractor(x)
        flattend = fmap.view(-1, 512)
        score = self._classifier(flattend)
        return score

    def _init_weight(self, layer):
        nn.init.kaiming_uniform_(layer.weight) if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) \
            else None




