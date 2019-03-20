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
from model.net import Vgg16

def train(cfgpath):

    augment = transforms.Compose([RandomHorizontalFlip(), RandomVerticalFlip(), ToTensor()])
    tr_ds = CIFAR10(root='./data', train=True, transform=augment, download=True)
    tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    val_ds = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)
    val_dl = DataLoader(val_ds, batch_size=64, num_workers=4)

