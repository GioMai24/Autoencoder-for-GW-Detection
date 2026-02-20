#!/usr/bin/env python
"""Optuna script to search for optimal hyperparameters. To be changed for each model as needed (multivariate version, see README)."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from tqdm import tqdm
import numpy as np
import optuna
import os
import gc
# custom
import custom.tools as tl
import custom.models as mod

optuna.logging.set_verbosity(optuna.logging.INFO)

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else: raise ValueError("cuda is not available?!")

print(f'Selected device: {device}')

## Useful parameters:
batch_size = 384
num_epochs = 20
verbosity= False
num_workers = 14 #
persistent_workers = False#True if num_workers>0 else False

main_dir = "/mnt/eph/multi_feat_model_stuff/"
h5file= "L1_70_12_6_6.h5"
h5_path = f"/mnt/eph/data/{h5file}"
## DB
optuna_url = f"sqlite:///{main_dir}optuna_study.db"
artifact_path = f"{main_dir}artifacts/"
artifact_store = optuna.artifacts.FileSystemArtifactStore(base_path=artifact_path)
trials_dir = f"{main_dir}trials/"

def objective(trial:optuna.Trial):
    num_epoch = num_epochs
    loss_func= nn.MSELoss()
    win_len = 48 # DOes not makes sense to tune win_len bc is structural of the net. trial.suggest_categorical('win_len', choices=[64,96, 128])

    win_stride = trial.suggest_categorical("win_stride", [0.875, 1.0])
    win_stride = int(win_len*win_stride)
    train = tl.h5set(h5_path,win_len=win_len, win_stride=win_stride, name='A') #50k rows
    val = tl.h5set(h5_path,win_len=win_len, win_stride=win_stride, name='B') #10k rows
    #test = h5set(h5_path,win_len=win_len, win_stride=win_stride, name='C') #10k rows
    train_loader = DataLoader(train,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                drop_last=True,
                )
    val_loader = DataLoader(val,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                drop_last=True,
                )
    num_layers =2

    # Recompute sequence length and input size for this dataset (depends on win_stride)
    sample = train.__getitem__(0)
    sq_len = sample.shape[0]


    ## autoencoder to device # did before.....
    autoencoder = mod.AEric2(
        sq_len=sq_len,
        num_feat=1*win_len,
        exp_dim=16*win_len,
        compr_dim=4*win_len,
        num_layers=num_layers,
        v=verbosity).to(device)
    autoencoder = torch.compile(autoencoder)

    optim_name = trial.suggest_categorical("optimizer", ['SGD', 'RMSprop', 'Adam', 'AdamW', 'NAdam'])
    clip = 2.
    decay = trial.suggest_categorical('decay', [0, 1e-6, 1e-4, 1e-5])
    
    if optim_name == 'SGD':
        momentum = trial.suggest_float("momentum", 0, 0.9, step=0.1)
        nesterov = trial.suggest_categorical('nesterov', [True, False]) if momentum > 0 else False
        lr = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)
        optim = getattr(torch.optim, optim_name)(autoencoder.parameters(), lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=decay)
    elif optim_name == 'RMSprop':
        momentum = trial.suggest_float("momentum", 0.3, 0.9, step=0.1)
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        optim = getattr(torch.optim, optim_name)(autoencoder.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    else:
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        optim = getattr(torch.optim, optim_name)(autoencoder.parameters(), lr=lr, weight_decay=decay)
    factor_lr = 0.98
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=factor_lr)


    losses= {'train':[], 'val':[]}
    scaler = GradScaler()
    for epoch in range(num_epoch):
        print(f"Starting epoch {epoch+1}/{num_epoch}")
        train_loss = tl.train_epoch(autoencoder, device, train_loader, loss_func, optim, scaler=scaler, clip=clip, multivariate=True)
        val_loss = tl.val_epoch(autoencoder, device, val_loader, loss_func, multivariate=True, scheduler=scheduler)
        trial.report(val_loss, epoch)
        losses['train'].append(train_loss)
        losses['val'].append(val_loss)
        print(f"Epoch {epoch+1}/{num_epoch}.. Train loss: {train_loss:.4f}.. Val loss: {val_loss:.4f}")

        if trial.should_prune():
            raise optuna.TrialPruned()

    # ensure trials directory exists and save the model there
    os.makedirs(trials_dir, exist_ok=True)
    save_path = f"{trials_dir}item2_{trial.number}.pt"
    tl.save_everything_multi(autoencoder,optim,train_loader, val_loader,scaler,losses, epoch, clip,scheduler, save_path)

    art_id = optuna.artifacts.upload_artifact(artifact_store = artifact_store, # Changed from optuna_url to artifact_store
                                            file_path = save_path,
                                            study_or_trial = trial.study)
                                            
    trial.set_user_attr("model_artifact_id", art_id)
    return val_loss


study = optuna.create_study(storage = optuna_url,
                            direction="minimize",
                            study_name="Item2", # Changed study name to create a new study
                            load_if_exists=True,
                            )
study.optimize(objective, n_trials=25, gc_after_trial=True, show_progress_bar=False)
trial = study.best_trial
print("Accuracy: {}".format(trial.value))
print("Best hyperparameters: {}".format(trial.params))
