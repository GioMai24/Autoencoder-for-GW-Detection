"""PyTorch Dataset class, and helper functions used in the scripts."""
import torch
from torch.utils.data import Dataset
import h5py as h5
import numpy as np
from tqdm import tqdm


class h5set(Dataset):
    """
    Custom Dataset class.

    Attributes
    ----------
    path : str
        Path to the file to initialize.
    name : str
        HDF5 dataset name to use.
    win_len : int
        Length of the windows the samples are split into.
    win_stride : int
        Separation between the windows.
    dataset : h5.File
        Read file.
    length : int
        Size of the dataset.
    """
    def __init__(self, path, win_len=None, win_stride=None, name='noise'):
        """
        Initialization.

        Parameters
        ----------
        path : str
            Path to the file to initialize.
        win_len : int
            Length of the windows the samples are split into.
        win_stride : int
            Separation between the windows.
        name : str
            HDF5 dataset name to use.
        """
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


def train_epoch(model, device, dataloader, loss_fn, optim, scaler, clip, multivariate):
    """
    Model training phase.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    device : torch.device
        Device to use.
    dataloader : torch.utils.data.Dataloader
        Dataloader to use.
    loss_fn : torch.nn.MSELoss
        Pytorch MSELoss module.
    optim : torch.optim
        PyTorch optimizer to use.
    scaler : torch.amp.scaler
        Pytorch scaler.
    clip : float
        Clipping applied to the parameters' gradients.
    multivariate : bool
        Flag to switch between univariate or multivariate mode.

    Returns
    -------
    epoch_loss / len(dataloader) : float
        Mean epoch loss.
    """
    model.train()
    epoch_loss = 0
    for batch_data in tqdm(dataloader):
        optim.zero_grad()
        # print(batch_data.element_size() * batch_data.nelement())
        batch_data = batch_data.to(device) if multivariate else batch_data.reshape(-1, 100, 1).to(device)
        # print(batch_data.shape)
        with torch.autocast(device_type=str(device), dtype=torch.float16):
            output = model(batch_data)
            loss = loss_fn(output, batch_data)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        if clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optim)
        scaler.update()
        # loss = np.sqrt(loss.item()) # if need
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)



def val_epoch(model, device, dataloader, loss_fn, multivariate, scheduler=None):
    """
    Model validation phase.

    Parameters
    ----------
    model : torch.nn.Module
        Model to validate.
    device : torch.device
        Device to use.
    dataloader : torch.utils.data.Dataloader
        Dataloader to use.
    loss_fn : torch.nn.MSELoss
        Pytorch MSELoss module.
    multivariate : bool
        Flag to switch between univariate or multivariate mode.
    scheduler : torch.optim.lr_scheduler
        LR scheduler to apply.

    Returns
    -------
    epoch_loss / len(dataloader) : float
        Mean epoch loss.
    """
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            batch_data = batch_data.to(device) if multivariate else batch_data.reshape(-1, 100, 1).to(device)
            with torch.autocast(device_type=str(device), dtype=torch.float16):
                output = model(batch_data)
                loss = loss_fn(output, batch_data)
            # loss = np.sqrt(loss.item())  # if need
            epoch_loss += loss.item()
    if scheduler is not None: scheduler.step()
    return epoch_loss / len(dataloader)



def test_function(model, device, dataloader):
    """
    Multivariate model test phase.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to test.
    device : torch.device
        Device to use.
    dataloader : torch.utils.data.Dataloader
        Dataloader to use.
        
    Returns
    -------
    all_window_losses : np.array
        Two dimensional numpy array. Rows are samples from the dataloader. Columns are time windows.
    """
    model.eval()
    loss_ew = torch.nn.MSELoss(reduction='none')
    #loss_tot = torch.nn.MSELoss(reduction='mean')
    losses = []
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            batch_data = batch_data.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16): output = model(batch_data)
            element_wise_loss = loss_ew(output, batch_data)
            segment_loss = element_wise_loss.mean(dim=2)
            losses.append(segment_loss.cpu())
        all_window_losses = torch.cat(losses, dim=0)
    return all_window_losses



def save_everything_uni(model, optimizer, train_loader, val_loader, scaler, losses, epoch, clip, save_path):
    """
    Save useful parameters using torch.save.
    UNIVARIATE MODE: no num_layers, and scheduler. See save_everything_multi.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model to save.
    optimizer : torch.optim
        PyTorch optimizer used to train the model.
    train_loader : torch.utils.data.Dataloader
        Dataloader used during the training phase.
    val_loader : torch.utils.data.Dataloader
        Dataloader used during the validation phase.
    scaler : torch.amp.scaler
        Scaler used during the training.
    losses : dict
        Dictionary of training and validation losses.
    epoch : int
        Last epoch of training.
    clip : float
        Clipping applied to the parameters' gradients
    save_path : str
        Path to the file to create.
    """
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
    # num_layers = model.Encoder.El1.num_layers

    checkpoint = { 'train_set_params': t_set_save,
                  'train_loader_params': tl_save,
                  'val_set_params': v_set_save,
                  'val_loader_params': vl_save,
                  'model_state_dict': model_state_dict,
                  'optimizer_state_dict': optimizer_state_dict,
                  'scaler_state_dict': scaler_state_dict,
                  'losses': losses,
                  'epoch':epoch,
                  'clip': clip}
                  # 'num_layers':num_layers}
    torch.save(checkpoint, save_path)



def save_everything_multi(model, optimizer, train_loader, val_loader, scaler, losses, epoch, clip, scheduler, save_path):
    """
    Save useful parameters using torch.save.
    MULTIVARIATE MODE. See save_everything_uni.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model to save.
    optimizer : torch.optim
        PyTorch optimizer used to train the model.
    train_loader : torch.utils.data.Dataloader
        Dataloader used during the training phase.
    val_loader : torch.utils.data.Dataloader
        Dataloader used during the validation phase.
    scaler : torch.amp.scaler
        Scaler used during the training.
    losses : dict
        Dictionary of training and validation losses.
    epoch : int
        Last epoch of training.
    clip : float
        Clipping applied to the parameters' gradients
    scheduler : torch.optim.lr_scheduler
        LR scheduler applied in the validation phase.
    save_path : str
        Path to the file to create.
    """
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

    scheduler_state_dict = scheduler.state_dict()

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
                  'num_layers':num_layers,
                  'scheduler_state_dict': scheduler_state_dict
                }
    torch.save(checkpoint, save_path)
