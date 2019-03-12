import os
from model.data import CIFAR100
from torch.utils.data import DataLoader


tr_filepath = os.path.join(os.getcwd(), 'data/train')
tr_ds = CIFAR100(tr_filepath)

tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True, drop_last=True, num_workers=4)
x, y = next(iter(tr_dl))


