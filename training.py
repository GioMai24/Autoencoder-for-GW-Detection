#!/usr/bin/env python

import torch 
import torch.nn as nn
import torch.optim as op
from torch.amp import GradScaler
from torch.utils.data import DataLoader
# custom
import custom.tools as tl
import custom.models as mod

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else: device = torch.device('cpu')
print(f'{device=}')

last_save = torch.load('/mnt/eph/data/optuna/artifacts/ba4a7fec-42c2-4b9f-a9cb-bc4f448671b1', weights_only=False)
save_path = '/mnt/eph/uni_feat_model_stuff/trial_20/'
num_epochs = 30
## Useful parameters:
batch_size = last_save['train_loader_params']['batch_size']
last_epoch = last_save['epoch']  # beware +1 convention
win_len = last_save['train_set_params']['win_len']
win_stride = last_save['train_set_params']['win_stride']
num_workers = 10
persistent_workers = True if num_workers>0 else False

h5_path = "/mnt/eph/data/L1_70_12_6_6.h5"
train = tl.h5set(h5_path, win_len=win_len, win_stride=win_stride, name='A')
val = tl.h5set(h5_path, win_len=win_len, win_stride=win_stride, name='B')

train_loader = DataLoader(train, batch_size=batch_size, pin_memory=True, num_workers=num_workers, persistent_workers=persistent_workers)
val_loader = DataLoader(val, batch_size=batch_size, pin_memory=True, num_workers=num_workers, persistent_workers=persistent_workers)


loss_func = nn.MSELoss()
model = mod.LSTMAEUni().to(device)
model = torch.compile(model)
model.load_state_dict(last_save['model_state_dict'])

# Trial choices
optim_name = 'NAdam'
lr = 0.0009419995643886475
clip = 4.964217914740257
decay = 0
if optim_name == 'SGD':
    momentum = 0
    nesterov = False
    optim = getattr(op, optim_name)(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=decay)
elif optim_name == 'RMSprop':
    momentum = 0
    optim = getattr(op, optim_name)(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
else: optim = getattr(op, optim_name)(model.parameters(), lr=lr, weight_decay=decay)
optim.load_state_dict(last_save['optimizer_state_dict'])

losses = last_save['losses']
losses['train'] = [loss.item() for loss in losses['train']]  # if old train_epoch function lacking .item()
scaler = GradScaler()
scaler.load_state_dict(last_save['scaler_state_dict'])
for epoch in range(last_epoch + 1, (tot_epochs := last_epoch + num_epochs + 1)):
    print(f'BEGIN EPOCH {epoch}/{tot_epochs - 1}')
    train_loss = tl.train_epoch(model, device, train_loader, loss_func, optim, scaler, clip, False)
    val_loss = tl.val_epoch(model, device, val_loader, loss_func, False)
    print(f'TRAIN - EPOCH {epoch}/{tot_epochs - 1} - loss: {train_loss} - test loss: {val_loss}')
    losses['train'].append(train_loss)
    losses['val'].append(val_loss)
    if not epoch % 5 or epoch == tot_epochs - 1:
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'scaler_state_dict': scaler.state_dict()}, f'{save_path}mod_opt_scal_{epoch}.pt')
        with open(f'{save_path}losses_{epoch}.pickle', 'wb') as fout: pickle.dump(losses, fout, protocol=-1)
