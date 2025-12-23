import torch
from torch.utils.data import Dataset
import h5py as h5

class h5set(Dataset):
    def __init__(self, path, win_len=None, win_stride=None, name='noise'):
        super().__init__()
        self.path = path
        self.name = name
        self.win_len = win_len
        self.win_stride = win_stride
        self.dataset = None
        with h5.File(path, mode="r", libver="latest", locking=False) as f:
            self.length = len(f[name])

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5.File(self.path, 'r', libver='latest', locking=False)
        item = torch.tensor(self.dataset[self.name][idx], dtype=torch.float32)
        return item.unfold(0, self.win_len, self.win_stride) if self.win_len else item




class h6set(Dataset):
    def __init__(self, path, win_len, win_stride, name='noise'):
        super().__init__()
        self.win_len = win_len
        self.win_stride = win_stride
        self.data = h5.File(path, locking=False)[name]
   
    def __len__(self): return len(self.data)

    def __getitem__(self, idx): return torch.tensor(self.data[idx], dtype=torch.float32).unfold(0, self.win_len, self.win_stride)

## 325k noise --> 25k test, 75k val, 225k train
# 25k test