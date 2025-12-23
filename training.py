from h5dataset import h5set
import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import pickle
# custom
from automobili import EricGio
from funkytrain import traingio

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

data = h5set('/mnt/eph/data/l1500mb_c.h5', 100, 30)
training = DataLoader(data, batch_size=1, num_workers=os.cpu_count())

AE = EricGio()
AE.to(device)
loss_func = nn.MSELoss()
lr = 1e-3
optim = torch.optim.Adam(AE.parameters(), lr=lr)



num_epochs = 10
losses = []
for epoch in range(num_epochs):
    ### Training (use the training function)
    train_loss = traingio(model=AE, device=device, dataloader=training, loss_fn=loss_func, optim=optim)
    print(f'TRAIN - EPOCH {epoch+1}/{num_epochs} - loss: {train_loss}')
    losses.append(train_loss)
with open('/mnt/eph/runs/stoopidsmalltrain.pickle', 'wb') as f: pickle.dump(losses, f, pickle.HIGHEST_PROTOCOL)