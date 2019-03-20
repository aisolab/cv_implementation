import os
import json
import fire
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from model.net import Vgg16
from tqdm import tqdm

def evaluate(cfgpath):
    # parsing json
    with open(os.path.join(os.getcwd(), 'experiments/config.json')) as io:
        params = json.loads(io.read())

    num_classes = params['model'].get('num_classes')
    batch_size = params['training'].get('batch_size')
    savepath = os.path.join(os.getcwd(), params['filepath'].get('ckpt'))
    ckpt = torch.load(savepath)

    # creating model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Vgg16(num_classes=num_classes)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    tr_ds = CIFAR10(root='./data', train=True, transform=ToTensor(), download=False)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, num_workers=4)
    val_ds = CIFAR10(root='./data', train=False, transform=ToTensor(), download=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

    tr_acc = 0
    for mb in tqdm(tr_dl, desc='iters'):
        x_mb, y_mb = map(lambda elm: elm.to(device), mb)

        with torch.no_grad():
            y_mb_hat = model(x_mb)
            y_mb_hat = torch.max(y_mb_hat, 1)[1]
            tr_acc += (y_mb_hat == y_mb).sum().item()
    else:
        tr_acc /= len(tr_ds)

    val_acc = 0
    for mb in tqdm(val_dl, desc='iters'):
        x_mb, y_mb = map(lambda elm: elm.to(device), mb)

        with torch.no_grad():
            y_mb_hat = model(x_mb)
            y_mb_hat = torch.max(y_mb_hat, 1)[1]
            val_acc += (y_mb_hat == y_mb).sum().item()
    else:
        val_acc /= len(val_ds)

    print('tr_acc: {:.2%}, val_acc: {:.2%}'.format(tr_acc, val_acc))

if __name__ == '__main__':
    fire.Fire(evaluate)