import torch
from torch.utils.data import Dataset
import pickle

class CIFAR100(Dataset):
    """CIFAR100 class"""
    def __init__(self, filepath: str) -> None:
        """Instantiatin CIFAR100 class

        Args:
            filepath: filepath
        """

        with open(filepath, mode='rb') as io:
           data_dict = pickle.load(io, encoding='bytes')

        keys = list(data_dict.keys())
        self.label = data_dict.get(keys[2])
        self.image = data_dict.get(keys[-1])

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        image = torch.tensor(self.image[[idx]].reshape(-1, 3, 32, 32)).float()
        label = torch.tensor(self.label[idx])
        return image, label