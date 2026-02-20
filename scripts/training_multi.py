#!/usr/bin/env python
"""Train the best models. To be changed for each run as needed (multivariate version, see README)."""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import h5py as h5
import numpy as np
from tqdm import tqdm
import pandas as pd
# custom
import custom.tools as tl
import custom.models as mod


def load_everything(pth_file, optimizer, device):
    """
    Load parameters saved with tl.save_everything_multi.
    Only multivariate mode.

    Parameters
    ----------
    pth_file : str
        Path to the file to load.
    optimizer : torch.optim
        Optimizer to initialize.
    device : torch.device
        Device to use.

    Returns
    -------
    train_set : h5set
        Initialized h5set training dataset.
    val_set : h5set
        Initialized h5set validation dataset.
    train_loader : torch.utils.data.Dataloader
        Initialized training Dataloader.
    val_loader : torch.utils.data.Dataloader
        Initialized validation Dataloader.
    model : Aeric2
        Initialized multivariate model.
    optimiz : torch.optim
        Initialized optimizer.
    scaler : torch.amp.scaler
        Initialzed training scaler.
    clip : float
        Clipping applied to the parameters' gradients.
    losses : dict
        Last training and validation losses.
    epoch : int
        Last epoch.
    scheduler : torch.optim.lr_scheduler
        Initialized validation LR scheduler.
    """
    saved = torch.load(pth_file, weights_only=False)
    
    path = saved['train_set_params']['path']
    dataset_name = saved['train_set_params']['name']
    win_len = saved['train_set_params']['win_len']
    win_stride = saved['train_set_params']['win_stride']
    train_set = tl.h5set(path,win_len=win_len, win_stride=win_stride, name=dataset_name) 

    path = saved['val_set_params']['path']
    dataset_name = saved['val_set_params']['name']
    win_len = saved['val_set_params']['win_len']
    win_stride = saved['val_set_params']['win_stride']
    val_set = tl.h5set(path,win_len=win_len, win_stride=win_stride, name=dataset_name)

    batch_size = saved['train_loader_params']['batch_size']
    num_workers = saved['train_loader_params']['num_workers']
    train_shuffle = True
    val_shuffle = False
    persistent_workers = True #saved['train_loader_params']['persistent_workers']
    pin_memory = saved['train_loader_params']['pin_memory']
    drop_last = saved['train_loader_params']['drop_last']
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=train_shuffle, pin_memory=pin_memory, num_workers=num_workers, persistent_workers=persistent_workers, drop_last=drop_last)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=val_shuffle, pin_memory=pin_memory, num_workers=num_workers, persistent_workers=persistent_workers, drop_last=drop_last)
    sq_len = train_set.__getitem__(0).shape[0]
    input_size =  train_set.__getitem__(0).shape[1]
    num_layers = saved['num_layers']
    clip = saved['clip']
    model = mod.AEric2(sq_len=sq_len, num_feat=1*input_size, exp_dim=16*input_size, compr_dim=4*win_len, num_layers=num_layers).to(device)
    model = torch.compile(model)
    model_state = saved['model_state_dict']
    model.load_state_dict(model_state)
    optimiz = optimizer(model.parameters())
    optimiz.load_state_dict(saved['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiz, gamma=0.98)
    if 'scheduler_state_dict' in saved:
        scheduler.load_state_dict(saved['scheduler_state_dict'])
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiz, mode='min', factor=0.5, patience=10)
    scaler = GradScaler()
    scaler.load_state_dict(saved['scaler_state_dict'])
    losses = saved['losses']
    epoch = saved['epoch']
    return train_set, val_set, train_loader, val_loader, model, optimiz, scaler, clip, losses, epoch, scheduler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Selected device: {device}')
load_path = './trials/item2_20.pt'
t_set, v_set, t_loader, v_loader, model, optimizer, scaler, clip, losses, epoch, scheduler = load_everything(load_path, torch.optim.AdamW, device)
num_epochs = 600
loss_func = nn.MSELoss()
best_loss = min(losses['val'])
for epoch in tqdm(range(epoch,num_epochs)):
    print(f"Starting epoch {epoch+1}/{num_epochs}")
    train_loss = tl.train_epoch(model, device, t_loader, loss_func, optimizer, scaler=scaler, clip=clip, multivariate=True)
    val_loss = tl.val_epoch(model, device, v_loader, loss_func, multivariate=True, scheduler=scheduler)
    losses['train'].append(train_loss)
    losses['val'].append(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs}.. Train loss: {train_loss:.4f}.. Val loss: {val_loss:.4f}")
    if val_loss < best_loss:
        best_loss = val_loss
        save_path = f'./trials/item2_20_epoch{epoch}.pt'
        tl.save_everything_multi(model, optimizer, t_loader, v_loader, scaler, losses, epoch, clip, scheduler, save_path)
