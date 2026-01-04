import torch
from torch.utils.data import Dataset
import h5py as h5
import numpy as np
from tqdm import tqdm


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
        return item.unfold(0, self.win_len, self.win_stride) if self.win_len is not None else item
## 325k noise --> 25k test, 75k val, 225k train
# 25k test


def train_epoch(model, device, dataloader, loss_fn, optim, scaler, clip, multivariate):
    model.train()
    epoch_loss = 0
    for batch_data in tqdm(dataloader):
        optim.zero_grad()
        # print(batch_data.element_size() * batch_data.nelement())
        batch_data = batch_data.to(device) if multivariate else batch_data.reshape(-1, 100, 1).to(device)
        # print(batch_data.shape)
        with torch.autocast(device=str(device), dtype=torch.float16):
            output = model(batch_data)
            loss = loss_fn(output, batch_data)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        if clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optim)
        scaler.update()
        # loss = np.sqrt(loss.item()) # if need
        epoch_loss += loss
    return epoch_loss / len(dataloader)



def val_epoch(model, device, dataloader, loss_fn, multivariate):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            batch_data = batch_data.to(device) if multivariate else batch_data.reshape(-1, 100, 1).to(device)
            with torch.autocast(device=str(device), dtype=torch.float16):
                output = model(batch_data)
                loss = loss_fn(output, batch_data)
            # loss = np.sqrt(loss.item())
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)
