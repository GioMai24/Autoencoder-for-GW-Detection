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



def save_everything(model,optimizer, train_loader, val_loader,scaler, losses,epoch,clip, save_path):
    train_set_params = train_loader.dataset.__dict__
    t_set_param_keep = ['path', 'name', 'win_len', 'win_stride']
    t_set_save = {k: train_set_params[k] for k in t_set_param_keep}

    train_loader_params = train_loader.__dict__
    tl_pkeep = ['batch_size','num_workers','pin_memory','drop_last','persistent_workers']
    tl_save = {k: train_loader_params[k] for k in tl_pkeep}

    val_set_params = val_loader.dataset.__dict__
    v_set_save = {k: val_set_params[k] for k in t_set_param_keep}

    val_loader_params = val_loader.__dict__
    vl_save = {k: val_loader_params[k] for k in tl_pkeep}

    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    scaler_state_dict = scaler.state_dict()
    num_layers = model.Encoder.El1.num_layers

    checkpoint = { 'train_set_params': t_set_save,
                  'train_loader_params': tl_save,
                  'val_set_params': v_set_save,
                  'val_loader_params': vl_save,
                  'model_state_dict': model_state_dict,
                  'optimizer_state_dict': optimizer_state_dict,
                  'scaler_state_dict': scaler_state_dict,
                  'losses': losses,
                  'epoch':epoch,
                  'clip': clip,
                  'num_layers':num_layers}
    torch.save(checkpoint, save_path)



def load_everything(pth_file, optimizer):
    saved = torch.load(pth_file, weights_only=False)
    
    path = saved['train_set_params']['path']
    dataset_name = saved['train_set_params']['name']
    win_len = saved['train_set_params']['win_len']
    win_stride = saved['train_set_params']['win_stride']
    train_set = h5set(path,win_len=win_len, win_stride=win_stride, name=dataset_name) 

    path = saved['val_set_params']['path']
    dataset_name = saved['val_set_params']['name']
    win_len = saved['val_set_params']['win_len']
    win_stride = saved['val_set_params']['win_stride']
    val_set = h5set(path,win_len=win_len, win_stride=win_stride, name=dataset_name)

    batch_size = saved['train_loader_params']['batch_size']
    num_workers = saved['train_loader_params']['num_workers']
    train_shuffle=True
    val_shuffle=False
    persistent_workers = saved['train_loader_params']['persistent_workers']
    pin_memory = saved['train_loader_params']['pin_memory']
    drop_last = saved['train_loader_params']['drop_last']
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=train_shuffle, pin_memory=pin_memory, num_workers=num_workers, persistent_workers=persistent_workers, drop_last=drop_last)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=val_shuffle, pin_memory=pin_memory, num_workers=num_workers, persistent_workers=persistent_workers, drop_last=drop_last)
    sq_len = train_set.__getitem__(0).shape[0]
    input_size =  train_set.__getitem__(0).shape[1]
    num_layers = saved['num_layers']
    clip = saved['clip']
    model = AEric(sq_len=sq_len, num_feat=1*input_size, exp_dim=8*input_size, compr_dim=8, num_layers=num_layers)
    model = torch.compile(model)
    model.load_state_dict(model_state)
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(saved['optimizer_state_dict'])
    scaler = GradScaler()
    scaler.load_state_dict(saved['scaler_state_dict'])
    losses = saved['losses']
    epoch = saved['epoch']
    return train_set, val_set, train_loader, val_loader, model, optimizer, scaler, clip, losses, epoch
