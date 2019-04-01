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
from tqdm import tqdm

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    avg_loss = 0

    for step, mb in tqdm(enumerate(dataloader), desc='steps', total=len(dataloader)):
        x_mb, y_mb = map(lambda elm: elm.to(device), mb)

        with torch.no_grad():
            mb_loss = loss_fn(model(x_mb), y_mb)
        avg_loss += mb_loss.item()
    else:
        avg_loss /= (step + 1)
    return avg_loss

def main(cfgpath):
    # parsing json
    with open(os.path.join(os.getcwd(), 'experiments/config.json')) as io:
        params = json.loads(io.read())

    num_classes = params['model'].get('num_classes')
    epochs = params['training'].get('epochs')
    batch_size = params['training'].get('batch_size')
    learning_rate = params['training'].get('learning_rate')

    # creating model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Vgg16(num_classes=num_classes)
    model.to(device)

    # creating dataset, dataloader
    augment = transforms.Compose([RandomHorizontalFlip(), RandomVerticalFlip(), ToTensor()])
    tr_ds = CIFAR10(root='./data', train=True, transform=augment, download=True)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_ds = CIFAR10(root='./data', train=False, transform=ToTensor(), download=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

    # training
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(params=model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(epochs), desc='epochs'):

        tr_loss = 0

        model.train()
        for step, mb in tqdm(enumerate(tr_dl), desc='steps', total=len(tr_dl)):
            x_mb, y_mb = map(lambda elm: elm.to(device), mb)

            opt.zero_grad()
            mb_loss = loss_fn(model(x_mb), y_mb)
            mb_loss.backward()
            opt.step()

            tr_loss += mb_loss.item()
        else:
            tr_loss /= (step + 1)

        val_loss = evaluate(model, val_dl, loss_fn, device)

        tqdm.write('epochs : {:3}, tr_loss: {:.3f}, val_loss: {:.3f}'.format(epoch + 1, tr_loss, val_loss))

    ckpt = {'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict()}

    savepath = os.path.join(os.getcwd(), params['filepath'].get('ckpt'))
    torch.save(ckpt, savepath)

if __name__ == '__main__':
    fire.Fire(main)

