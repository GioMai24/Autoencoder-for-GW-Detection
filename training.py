from h5dataset import h5set
import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
# custom
from automobili import EricGio
from funkytrain import traingio, evalgio

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# data = h5set('/mnt/eph/data/l1500mb_c.h5', 100, 30)
data_train = torch.load("/mnt/eph/data/train_test_val/train.pt", weights_only=False)
training = DataLoader(data_train, batch_size=1, num_workers=os.cpu_count())

data_val = torch.load("/mnt/eph/data/train_test_val/val.pt", weights_only=False)
validation = DataLoader(data_val, batch_size=1, num_workers=os.cpu_count())

model = EricGio()
model.to(device)
loss_func = nn.MSELoss()
lr = 1e-3
optim = torch.optim.Adam(model.parameters(), lr=lr)

num_epochs = 100
losses = {'train':[], 'test':[]}
for epoch in range(1, num_epochs + 1):
    train_loss = traingio(model=model, device=device, dataloader=training, loss_fn=loss_func, optim=optim)
    losses['train'].append(train_loss)
    val_loss = evalgio(model=model, device=device, dataloader=validation, loss_fn=loss_func)
    losses['test'].append(val_loss)
    print(f'TRAIN - EPOCH {epoch}/{num_epochs} - loss: {train_loss} - test loss: {val_loss}')
    if epoch % 10 == 0 or epoch == num_epochs:
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optim.state_dict()}, f'/mnt/eph/runs/model_optim_weights/model_optim_{epoch}.pt')
        with open(f'/mnt/eph/runs/losses/loss_{epoch}.pickle', 'wb') as fout: pickle.dump(losses, fout, protocol=-1)
