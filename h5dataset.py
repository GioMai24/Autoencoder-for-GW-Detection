import torch
from torch.utils.data import Dataset
import h5py as h5

class h5set(Dataset):
    def __init__(self, path, dataset='noise'):
        super().__init__()
        self.path = path
        self.name = dataset
        self.dataset = None
        with h5.File(path, mode="r", libver="latest", locking=False) as f:
            self.length = len(f[dataset])

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5.File(self.path, 'r', libver='latest', locking=False)
        item = self.dataset[self.name][idx]
        return torch.tensor(item, dtype=torch.float32)

## 325k noise --> 25k test, 75k val, 225k train
# 25k test