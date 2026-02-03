#!/usr/bin/env python

import os
import sys
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as op
from torch.amp import GradScaler
import optuna
import logging
import pickle
from tqdm import tqdm
# custom
import custom.models as mod
import custom.tools as tl

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else: raise ValueError("cuda is not available?!")
print(f'{device=}')

## Useful parameters:
batch_size = 64
num_epochs = 30
win_len = 100
win_stride = 60
num_workers = 10
persistent_workers = True if num_workers>0 else False

main_dir = "/mnt/eph/data/optuna/"
h5_path = "/mnt/eph/data/L1_70_12_6_6.h5"
train = tl.h5set(h5_path, win_len=win_len, win_stride=win_stride, name='A')
val = tl.h5set(h5_path, win_len=win_len, win_stride=win_stride, name='B')

train_loader = DataLoader(train, batch_size=batch_size, pin_memory=True, num_workers=num_workers, persistent_workers=persistent_workers)
val_loader = DataLoader(val, batch_size=batch_size, pin_memory=True, num_workers=num_workers, persistent_workers=persistent_workers)

## DB
optuna_url = f"sqlite:///{main_dir}optuna_study.db"
artifact_path = f"{main_dir}artifacts/"
artifact_store = optuna.artifacts.FileSystemArtifactStore(base_path=artifact_path)

def objective(trial:optuna.Trial):
    # Trial choices
    dropout = trial.suggest_float("dropout", 0, 0.75, step=0.25)
    optim_name = trial.suggest_categorical("optimizer", ['Adam', 'SGD', 'RMSprop', 'NAdam'])
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    clip = trial.suggest_float('clip', .2, 5)
    decay = trial.suggest_categorical('decay', [0, 1e-4, 1e-3, 1e-2, .1])
    
    loss_func = nn.MSELoss()
    model = mod.DeepAE(h_s1=128, n_l1=2, h_s2=32, n_l2=2, h_s3=8, n_l3=3, dropout=dropout).to(device)
    model = torch.compile(model)
        
    if optim_name == 'SGD':
        momentum = trial.suggest_float("momentum", 0, 0.9, step=0.1)
        nesterov = trial.suggest_categorical('nesterov', [True, False]) if momentum > 0 else False
        optim = getattr(op, optim_name)(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=decay)
    elif optim_name == 'RMSprop':
        momentum = trial.suggest_float("momentum", 0.3, 0.9, step=0.1)
        optim = getattr(op, optim_name)(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    else: optim = getattr(op, optim_name)(model.parameters(), lr=lr, weight_decay=decay)
    
    losses = {'train':[], 'val':[]}
    scaler = GradScaler()
    for epoch in range(1, num_epochs + 1):
        train_loss = tl.train_epoch(model, device, train_loader, loss_func, optim, scaler, clip, False)
        val_loss = tl.val_epoch(model, device, val_loader, loss_func, False)
        print(f'TRAIN - EPOCH {epoch}/{num_epochs} - loss: {train_loss} - test loss: {val_loss}')
        trial.report(val_loss, epoch)
        if trial.should_prune(): raise optuna.TrialPruned()
        losses['train'].append(train_loss)
        losses['val'].append(val_loss)

    save_path = f'{main_dir}checkpoint.pt'
    tl.save_everything(model, optim, train_loader, val_loader, scaler, losses, epoch, clip, save_path)

    art_id = optuna.artifacts.upload_artifact(artifact_store=artifact_store, file_path=save_path, study_or_trial=trial.study)
    trial.set_user_attr('everything_id', art_id)
    return val_loss

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
# to change model, just change study_name, don't touch storage
study = optuna.create_study(direction='minimize', study_name='Deep1282_322_83', storage=optuna_url, load_if_exists=True)
study.optimize(objective, n_trials=30)
