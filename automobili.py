import os
import torch 
import torch.nn as nn


class Encoder_Moreno(nn.Module):
    def __init__(self, num_feat, exp_dim, compr_dim, num_layers, v=False):
        super().__init__()
        ## Useful quantities
        self.v = v
        self.El1 = nn.LSTM(input_size=num_feat, 
                           hidden_size=exp_dim,
                           num_layers=num_layers,
                           batch_first=True)
        self.El2 = nn.LSTM(input_size=exp_dim,
                           hidden_size=compr_dim, 
                           num_layers=num_layers,
                           batch_first=True)

    
    def forward(self, item):
        sq_len = item.shape[1]  # to be changed if batch_first=False
        if self.v: 
            print(item.shape, "Input shape")
            item, _ = self.El1(item)
            print(item.shape, "1st encoder layer output shape")
            item, _ = self.El2(item)
            print(item.shape, "2nd encoder layer output shape")
            item = item[:,-1,:]
            print(item.shape, "return sequence= False analog")
            item = item.repeat(1, sq_len, 1)
            print(item.shape, "repeat vector 100x")
            return item
        else:
            item, h_c = self.El1(item)
            item, h_c = self.El2(item)
            item = item[:,-1,:]
            item = item.repeat(1, sq_len,1)
            return item

class Decoder_Moreno(nn.Module):
    def __init__(self, num_feat, exp_dim, compr_dim, num_layers, v=False):
        super().__init__()
        self.v = v
        self.Dl1 = nn.LSTM(input_size=compr_dim, 
                           hidden_size=compr_dim,
                           num_layers=num_layers,
                           batch_first=True)
        self.Dl2 = nn.LSTM(input_size=compr_dim,
                           hidden_size=exp_dim,
                           num_layers=num_layers,
                           batch_first=True)
        self.TimeDistributed = nn.Conv1d(exp_dim,
                                        num_feat,
                                        kernel_size=1)

        
    def forward(self, item):
        if self.v: 
            item, h_c = self.Dl1(item)
            print(item.shape, "1st decoder layer output shape")
            item, h_c = self.Dl2(item)
            print(item.shape, "2nd decoder layer output shape")
            item = torch.movedim(item, 1,2)
            print(item.shape, "move dim shape")
            item = self.TimeDistributed(item)
            print(item.shape, "conv1d shape (time distributed)")
            return item
        else:
            item, _ = self.Dl1(item)
            item, _ = self.Dl2(item)
            item = torch.movedim(item, 1,2)
            item = self.TimeDistributed(item)
            return item

class AEric(nn.Module):
    def __init__(self, num_feat, exp_dim, compr_dim, num_layers, v=False):
        super().__init__()
        self.Encoder = Encoder_Moreno(num_feat, exp_dim, compr_dim, num_layers, v=v)
        self.Decoder = Decoder_Moreno(num_feat, exp_dim, compr_dim, num_layers, v=v)

    
    def forward(self, item):
        item = self.Encoder(item)
        item = self.Decoder(item)
        return item
        # encoded = self.Encoder(item)
        # decoded = self.Decoder(encoded)
        # return decoded



class EricGio(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.El1 = nn.LSTM(input_size=1, hidden_size=32, num_layers=3, batch_first=True)
        self.El2 = nn.LSTM(input_size=32, hidden_size=8, num_layers=3, batch_first=True)  
        self.Dl1 = nn.LSTM(input_size=8, hidden_size=8, num_layers=3, batch_first=True)
        self.Dl2 = nn.LSTM(input_size=8, hidden_size=32, num_layers=3, batch_first=True)
        self.TimeDistributed = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1)  # Not sure it corresponds to Keras TimeDistributed(Dense)

    def forward(self, x):
        # Encoding
        print(f'Input {x.shape=}')
        x, _ = self.El1(x)  # Send whole output (corresponds to 100x32 in paper)
        print(f"First LSTM out {x.shape=}")
        
            ## gg El2
        x, _ = self.El2(x)  ## gg method
        print(f"Second LSTM out {x[:,-1,:].shape=}")

            ## gio El2
        # _, (x, _) = self.El2(x)  # Send last output (corresponds to whole[-1], has 1 value for each layer (3)) I guess the paper takes only the last of these x[-1]
        # print(f"Second LSTM out {x[-1].shape=}")

        # Repeating
        # x = x[-1].unsqueeze(1).repeat(1, 100, 1)  # Tensor.unsqueeze(x) adds a dimension to x position, to have batch dim back.  ## gio method
        x = x[:, -1, :].repeat(1, 100, 1)  ## gg method
        print(f"Repeated {x.shape=}")

        # Decoding
        print("Decoding")
        x, _ = self.Dl1(x)
        print(f"First LSTM out {x.shape=}")
        x, _ = self.Dl2(x)
        print(f"Second LSTM out {x.shape=}")
        x = torch.movedim(x, 1, 2)
        print(f"3D transposed {x.shape=}")
        x = self.TimeDistributed(x)
        print(f'Convoluted {x.shape=}')
        x = torch.movedim(x, 1, 2)
        print(f'Back to original dim {x.shape=}')
        return x