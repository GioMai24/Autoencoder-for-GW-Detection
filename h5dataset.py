import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py as h5
import os


class h5set(torch.utils.data.Dataset):
    def __init__(self, path, dataset='noise'):
        super().__init__()
        self.path = path
        self.dataset = None
        with h5.File(name=self.path,mode="r", libver="latest", locking=False) as f:
            self.length = len(f[dataset])

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.dataset==None:
            self.dataset= h5.File(self.path, 'r', libver='latest', locking=False)
        item = self.dataset[dataset][idx]
        return torch.from_numpy(item)



## 325k noise --> 25k test, 75k val, 225k train
# 25k test