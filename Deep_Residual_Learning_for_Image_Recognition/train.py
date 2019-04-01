import os
import json
import fire
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, ToTensor
from torch.utils.data import DataLoader
from model.net import ResNet50
from model.net import Bottleneck
from tqdm import tqdm
