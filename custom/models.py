import os
import torch 
import torch.nn as nn


class LSTMAEUni(nn.Module):
    def __init__(self):
        super().__init__()
        self.El1 = nn.LSTM(input_size=1, hidden_size=32, num_layers=3, batch_first=True)
        self.El2 = nn.LSTM(input_size=32, hidden_size=8, num_layers=3, batch_first=True)  
        self.Dl1 = nn.LSTM(input_size=8, hidden_size=8, num_layers=3, batch_first=True)
        self.Dl2 = nn.LSTM(input_size=8, hidden_size=32, num_layers=3, batch_first=True)
        self.TimeDistributed = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1)

    def forward(self, x):
        ## Encoding
        # print(f'Input {x.shape=}')
        x, _ = self.El1(x)  # Send whole output (corresponds to 100x32 in paper)
        # print(f"First LSTM out {x.shape=}")
        _, (x, _) = self.El2(x)  # Send last output (corresponds to whole[-1], has 1 value for each layer (3)) I guess the paper takes only the last of these x[-1]
        # print(f"Second LSTM out {x[-1].shape=}")

        ## Repeating
        x = x[-1].unsqueeze(1).repeat(1, 100, 1)  # Tensor.unsqueeze(x) adds a dimension to x position, to have batch dim back.  ## gio method
        # print(f"Repeated {x.shape=}")

        ## Decoding
        # print("Decoding")
        x, _ = self.Dl1(x)
        # print(f"First LSTM out {x.shape=}")
        x, _ = self.Dl2(x)
        # print(f"Second LSTM out {x.shape=}")
        x = torch.movedim(x, 1, 2)
        # print(f"3D transposed {x.shape=}")
        x = self.TimeDistributed(x)
        # print(f'Convoluted {x.shape=}')
        x = torch.movedim(x, 1, 2)
        # print(f'Back to original dim {x.shape=}')
        return x