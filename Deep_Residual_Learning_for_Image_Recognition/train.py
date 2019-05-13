import json
import fire
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, ToTensor, Normalize
from torch.utils.data import DataLoader
from model.net import ResNet50
from tensorboardX import SummaryWriter
from tqdm import tqdm


def evaluate(model, dataloader, loss_fn, device):
    if model.training:
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


def main(cfgpath, global_step):
    proj_dir = Path.cwd()
    # parsing json
    with open(proj_dir / cfgpath) as io:
        params = json.loads(io.read())

    num_classes = params['model'].get('num_classes')
    epochs = params['training'].get('epochs')
    batch_size = params['training'].get('batch_size')
    learning_rate = params['training'].get('learning_rate')

    # creating model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ResNet50(num_classes=num_classes)
    model.to(device)

    # creating dataset, dataloader
    transform_tr = transforms.Compose([RandomCrop(32, 4),
                                       RandomHorizontalFlip(),
                                       ToTensor(),
                                       Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_val = transforms.Compose([ToTensor(),
                                        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    tr_ds = CIFAR10(root='./data', train=True, transform=transform_tr, download=True)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_ds = CIFAR10(root='./data', train=False, transform=transform_val, download=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

    # training
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
    writer = SummaryWriter('./runs/exp')

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
            if (epoch * len(tr_dl) + step) % global_step == 0:
                val_loss = evaluate(model, val_dl, loss_fn, device)
                writer.add_scalars('loss', {'train': tr_loss / (step + 1),
                                            'validation': val_loss}, epoch * len(tr_dl) + step)
                model.train()

        else:
            tr_loss /= (step + 1)
        val_loss = evaluate(model, val_dl, loss_fn, device)
        scheduler.step(val_loss)
        tqdm.write('epochs : {:3}, tr_loss: {:.3f}, val_loss: {:.3f}'.format(epoch + 1, tr_loss, val_loss))

    ckpt = {'epoch': epoch,
            'model_state_dict': model.state_dict()}

    savepath = proj_dir / params['filepath'].get('ckpt')
    torch.save(ckpt, savepath)


if __name__ == '__main__':
    fire.Fire(main)