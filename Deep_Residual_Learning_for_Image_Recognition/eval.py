import json
import fire
import torch
from pathlib import Path
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize
from torch.utils.data import DataLoader
from model.net import ResNet50
from tqdm import tqdm


def get_accuracy(model, dataloader, device):
    correct_count = 0
    total_count = 0
    for mb in tqdm(dataloader, desc='steps'):
        x_mb, y_mb = map(lambda elm: elm.to(device), mb)

        with torch.no_grad():
            y_mb_hat = torch.max(model(x_mb), 1)[1]
            correct_count += (y_mb_hat == y_mb).sum().item()
            total_count += x_mb.size()[0]
    else:
        acc = correct_count / total_count
    return acc


def main(cfgpath):
    proj_dir = Path.cwd()
    # parsing json
    with open(proj_dir / cfgpath) as io:
        params = json.loads(io.read())

    num_classes = params['model'].get('num_classes')
    batch_size = params['training'].get('batch_size')
    savepath = proj_dir / params['filepath'].get('ckpt')
    ckpt = torch.load(savepath)

    # creating model

    model = ResNet50(num_classes=num_classes)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # creating dataset, dataloader
    transform = transforms.Compose([ToTensor(),
                                    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    tr_ds = CIFAR10(root='./data', train=True, transform=transform, download=False)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, num_workers=4)
    val_ds = CIFAR10(root='./data', train=False, transform=transform, download=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

    # evaluate
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    tr_acc = get_accuracy(model, tr_dl, device)
    val_acc = get_accuracy(model, val_dl, device)
    print('tr_acc: {:.2%}, val_acc: {:.2%}'.format(tr_acc, val_acc))

if __name__ == '__main__':
    fire.Fire(main)
