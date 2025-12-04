import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py as h5
import os


class h5set(torch.utils.data.Dataset):
    def __init__(self, path, train=True):
        super().__init__()
        self.path = path
        self.dataset = None
        with h5.File(name=self.path,mode="r") as f:
            self.groups = list(f.keys())
            self.single_lengths = [len(f[g]) for g in self.groups]
            if train:
                self.length = len(f[self.groups[1]])
            else:
                self.length = len(f[self.groups[0]])

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.dataset==None:
            self.dataset= h5.File(self.path, 'r', libver='latest', locking=False)
        if train:
            item = self.dataset[self.groups[1]][idx]
        else:
            item = self.dataset[self.groups[0]][idx]
        return torch.from_numpy(item)
