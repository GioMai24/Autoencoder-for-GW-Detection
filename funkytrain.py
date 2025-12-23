import numpy as np

def traingio(model, device, dataloader, loss_fn, optim):
    model.train()
    epoch_loss = 0
    for batch_data in dataloader:
        optim.zero_grad()
        # print(batch_data.element_size() * batch_data.nelement())
        batch_data = batch_data.reshape(-1, 100, 1).to(device)
        # print(batch_data.shape)
        output = model(batch_data)
        loss = loss_fn(output, batch_data)
        loss.backward()
        optim.step()
        # loss = np.sqrt(loss.item()) # if need
        epoch_loss += loss
    return epoch_loss / len(dataloader)



def train_epoch(ae, device, dataloader,timestep, loss_fn, optim):
    ae.train()
    losses = []
    for batch_data in dataloader:
        num_time_steps = batch_data.shape[1]
        remainder = num_time_steps % timestep
        batch_data = batch_data[:, :-remainder]
        for sample_idx in range(batch_data.shape[0]):
            sequence = batch_data[sample_idx, :]
            segments = sequence.reshape(-1, timestep, 1)
            for segment_idx in range(segments.shape[0]):
                current_window = segments[segment_idx, :, :]
                c_w = current_window.unsqueeze(0)
                c_w = c_w.to(device)
                ae_output = ae(c_w)
                loss = loss_fn(ae_output,c_w)
                print(loss)
                optim.zero_grad()
                loss.backward()
                optim.step()

                losses.append(loss.detach().cpu().numpy())
    losses = np.mean(losses)
    return losses