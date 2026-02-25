"""Autoencoder models built with PyTorch modules."""
import os
import torch 
import torch.nn as nn


class LSTMAEUni(nn.Module):
    """
    Original Moreno AE as described in 'https://arxiv.org/abs/2107.12698'.

    Attributes
    ----------
    El* : nn.LSTM
        Encoding layers.
    Dl* : nn.LSTM
        Decoding layers.
    TimeDistributed : nn.Conv1d
        Mimics keras.layers.TimeDistributed behaviour.
    """
    
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
        _, (x, _) = self.El2(x)  # Send last output (corresponds to whole[-1], has 1 value for each layer (3))
        # print(f"Second LSTM out {x[-1].shape=}")

        ## Repeating
        x = x[-1].unsqueeze(1).repeat(1, 100, 1)  # Tensor.unsqueeze(x) adds a dimension to x position, to have batch dim back.
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



class DeepAE(nn.Module):
    """
    As LSTMAEuni, but deeper.

    Attributes
    ----------
    El* : nn.LSTM
        Encoding layers.
    Dl* : nn.LSTM
        Decoding layers.
    TimeDistributed : nn.Conv1d
        Mimics keras.layers.TimeDistributed behaviour.
    """
    
    def __init__(self, h_s1, n_l1, h_s2, n_l2, h_s3, n_l3, dropout, reg = True):
        """
        Initialization.

        Parameters
        ----------
        h_s* : int
            Hidden feature dimension size of the *-th LSTM block.
        n_l* : int
            Number of layers of the *-th LSTM block.
        dropout : float
            Dropout applied to each LSTM block.
        """
        super().__init__()
        self.El1 = nn.LSTM(input_size=1, hidden_size=h_s1, num_layers=n_l1, batch_first=True, dropout=dropout)
        self.El2 = nn.LSTM(input_size=h_s1, hidden_size=h_s2, num_layers=n_l2, batch_first=True, dropout=dropout)
        self.El3 = nn.LSTM(input_size=h_s2, hidden_size=h_s3, num_layers=n_l3, batch_first=True, dropout=dropout)
        
        self.Dl1 = nn.LSTM(input_size=h_s3, hidden_size=h_s3, num_layers=n_l3, batch_first=True, dropout=dropout)
        self.Dl2 = nn.LSTM(input_size=h_s3, hidden_size=h_s2, num_layers=n_l2, batch_first=True, dropout=dropout)
        self.Dl3 = nn.LSTM(input_size=h_s2, hidden_size=h_s1, num_layers=n_l1, batch_first=True, dropout=dropout)
        self.TimeDistributed = nn.Conv1d(in_channels=h_s1, out_channels=1, kernel_size=1)
        self.reg = reg

    def forward(self, x):
        ## Encoding
        # print(f'Input {x.shape=}')
        x, _ = self.El1(x)
        # print(f"First LSTM out {x.shape=}")
        x, _ = self.El2(x)
        # print(f"Second LSTM out {x.shape=}")
        _, (x, _) = self.El3(x)
        # print(f"Third LSTM out {x[-1].shape=}")
        
        ## Repeating
        x = x[-1].unsqueeze(1).repeat(1, 100, 1)  # Tensor.unsqueeze(x) adds a dimension to x position, to have batch dim back.
        # print(f"Repeated {x.shape=}")

        # ## Decoding
        # print("Decoding")
        x, _ = self.Dl1(x)
        # print(f"First LSTM out {x.shape=}")
        x, _ = self.Dl2(x)
        # print(f"Second LSTM out {x.shape=}")
        x, _ = self.Dl3(x)
        # print(f"Third LSTM out {x.shape=}")
        x = torch.movedim(x, 1, 2)
        # print(f"3D transposed {x.shape=}")
        x = self.TimeDistributed(x)
        # print(f'Convoluted {x.shape=}')
        x = torch.movedim(x, 1, 2)
        # print(f'Back to original dim {x.shape=}')
        if self.reg: x = nn.Sigmoid()(x)  
        return x



class ResAE(nn.Module):
    """
    As LSTMAEuni, with skip connections.

    Attributes
    ----------
    El* : nn.LSTM
        Encoding layers.
    Dl* : nn.LSTM
        Decoding layers.
    TimeDistributed : nn.Conv1d
        Mimics keras.layers.TimeDistributed behaviour.
    """
    
    def __init__(self, dropout, reg):
        """
        Initialization

        Parameters
        ----------
        dropout : float
            Dropout applied to each LSTM block.
        """
        super().__init__()
        self.El1 = nn.LSTM(input_size=1, hidden_size=32, num_layers=3, batch_first=True, dropout=dropout)
        self.El2 = nn.LSTM(input_size=32, hidden_size=8, num_layers=3, batch_first=True, dropout=dropout)  
        self.Dl1 = nn.LSTM(input_size=8, hidden_size=8, num_layers=3, batch_first=True, dropout=dropout)
        self.Dl2 = nn.LSTM(input_size=8, hidden_size=32, num_layers=3, batch_first=True, dropout=dropout)
        self.TimeDistributed = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1)
        self.reg = reg

    def forward(self, x):
        ## Encoding
        # print(f'Input {inp.shape=}')
        x, _ = self.El1(x)  # Send whole output (corresponds to 100x32 in paper)
        # res1 = x.clone()  # save to skip
        # print(f"First LSTM out {x.shape=}")
        _, (x, _) = self.El2(x)  # Send last output (corresponds to whole[-1], has 1 value for each layer (3)) I guess the paper takes only the last of these x[-1]
        # print(f"Second LSTM out {x[-1].shape=}")

        ## Repeating
        x = x[-1].unsqueeze(1).repeat(1, 100, 1)  # Tensor.unsqueeze(x) adds a dimension to x position, to have batch dim back.
        # print(f"Repeated {x.shape=}")
        res2 = x.clone()  # save to skip

        ## Decoding
        # print("Decoding")
        x, _ = self.Dl1(x)
        # print(f"First LSTM out {x.shape=}")
        x = x + res2  # skip
        x, _ = self.Dl2(x)
        # print(f"Second LSTM out {x.shape=}")
        # x = x + res1  # skip
        x = torch.movedim(x, 1, 2)
        # print(f"3D transposed {x.shape=}")
        x = self.TimeDistributed(x)
        # print(f'Convoluted {x.shape=}')
        x = torch.movedim(x, 1, 2)
        # print(f'Back to original dim {x.shape=}')
        if self.reg == True: x = nn.Sigmoid()(x)
        return x



class Encoder_Moreno(nn.Module):
    """
    Encoder module for the multivariate implementation Aeric2 below.

    Attributes
    ----------
    v : bool
        Activate verbosity to check layers' output.
    sq_len : int
        Length of the input sequence.
    El* : nn.LSTM
        Encoding layers.
    ln* : nn.LayerNorm
        Normalization layers.
    """
    
    def __init__(self, sq_len, num_feat, exp_dim, compr_dim, num_layers, v=False):
        """
        Initialization.

        Parameters
        ----------
        sq_len : int
            Length of the input sequence.
        num_feat : int
            Feature dimension of the input sequence.
        exp_dim : int
            Hidden feature dimension size of the first LSTM block.
        compr_dim : int
            Hidden feature dimension size of the last LSTM block.
        num_layers : int
            Number of layers of each LSTM block.
        v : bool
            Verbosity.            
        """
        super().__init__()
        self.v = v
        self.sq_len = sq_len
        self.El1 = nn.LSTM(input_size=num_feat, 
                           hidden_size=exp_dim,
                           num_layers=num_layers,
                           batch_first=True)
        self.ln1 = nn.LayerNorm(exp_dim) 
        self.El2 = nn.LSTM(input_size=exp_dim,
                           hidden_size=compr_dim, 
                           num_layers=num_layers,
                           batch_first=True)
        self.ln2 = nn.LayerNorm(compr_dim)

           
    def forward(self, item):
        if self.v: 
            print(item.shape, "Input shape")
            item, _ = self.El1(item)
            item = self.ln1(item)
            print(item.shape, "1st encoder layer output shape")
            item, (h_n,c_n) = self.El2(item)
            item = self.ln2(item)
            print(item.shape, "2nd encoder layer output shape")
            item = item[:,-1,:]#torch.mean(item, dim=1) # too aggressive? seq2seq uses item[:,-1;:]
            print(item.shape, "return sequence= False analog")
            item = item[:,None,:]
            item = item.repeat(1, self.sq_len, 1)
            print(item.shape, "repeat vector sq_len times")
            return item, (h_n,c_n)
        else:
            item, _ = self.El1(item)
            item = self.ln1(item)
            item, (h_n,c_n) = self.El2(item)
            item = self.ln2(item)
            item = torch.mean(item, dim=1)
            item = item[:,None,:]
            item = item.repeat(1, self.sq_len,1)
            return item, (h_n,c_n)


class Decoder_Moreno(nn.Module):
    """
    Decoder module for the multivariate implementation Aeric2 below.

    Attributes
    ----------
    sq_len : int
        Length of the input sequence.
    v : bool
        Activate verbosity to check layers' output.
    Dl* : nn.LSTM
        Encoding layers.
    ln* : nn.LayerNorm
        Normalization layers.
    TimeDistributed : nn.Conv1d
        Mimics keras.layers.TimeDistributed behaviour.
    """
    
    def __init__(self, sq_len, num_feat, exp_dim, compr_dim, num_layers, v=False):
        """
        Initialization.

        Parameters
        ----------
        sq_len : int
            Length of the input sequence.
        num_feat : int
            Feature dimension of the input sequence.
        exp_dim : int
            Hidden feature dimension size of the first LSTM block.
        compr_dim : int
            Hidden feature dimension size of the last LSTM block.
        num_layers : int
            Number of layers of each LSTM block.
        v : bool
            Verbosity.            
        """
        super().__init__()
        self.v = v
        self.sq_len = sq_len
        self.Dl1 = nn.LSTM(input_size=compr_dim, 
                           hidden_size=compr_dim,
                           num_layers=num_layers,
                           batch_first=True)
        self.ln1 = nn.LayerNorm(compr_dim)
        self.Dl2 = nn.LSTM(input_size=compr_dim,
                           hidden_size=exp_dim,
                           num_layers=num_layers,
                           batch_first=True)
        self.TimeDistributed = nn.Conv1d(exp_dim,
                                        num_feat,
                                        kernel_size=1)

        
    def forward(self, item, encoder_state):
        if self.v: 
            item, _ = self.Dl1(item, encoder_state)
            item = self.ln1(item)
            print(item.shape, "1st decoder layer output shape")
            item, _ = self.Dl2(item)
            print(item.shape, "2nd decoder layer output shape")
            item = torch.movedim(item, 1,2)
            print(item.shape, "move dim shape")
            item = self.TimeDistributed(item)
            print(item.shape, "conv1d shape (time distributed)")
            item = torch.movedim(item, 1,2)
            print(item.shape, "final output shape")
            return item
        else:
            item, _ = self.Dl1(item, encoder_state)
            item = self.ln1(item)
            item, _ = self.Dl2(item)
            item = torch.movedim(item, 1,2)
            item = self.TimeDistributed(item)
            item = torch.movedim(item, 1,2)
            return item


class AEric2(nn.Module):
    """
    Multivariate autoencoder implementation.
    
    Attributes
    ----------
    Encoder : Encoder_Moreno
        Encoder network.
    Decoder : Decoder_Moreno
        Decoder network.
    """
    
    def __init__(self,sq_len, num_feat, exp_dim, compr_dim, num_layers, v=False):
        """
        Initialization.

        Parameters
        ----------
        sq_len : int
            Length of the input sequence.
        num_feat : int
            Feature dimension of the input sequence.
        exp_dim : int
            Hidden feature dimension size of the first LSTM block.
        compr_dim : int
            Hidden feature dimension size of the last LSTM block.
        num_layers : int
            Number of layers of each LSTM block.
        v : bool
            Verbosity.            
        """
        super().__init__()
        self.Encoder = Encoder_Moreno(sq_len, num_feat, exp_dim, compr_dim, num_layers, v=v)
        self.Decoder = Decoder_Moreno(sq_len, num_feat, exp_dim, compr_dim, num_layers, v=v)
        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization from: 'https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization'.
        Set forget gate to remember more things. Base LSTM init is inversely proportional to hidden size.
        (All the weights and biases are initialized from U(-sqrt(k),sqrt(k)) where k=1/hidden_size)
        """
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'TimeDistributed' in name:
                if 'weight' in name:
                    nn.init.kaiming_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)
    
    def forward(self, item):
        encoded, hidden_state = self.Encoder(item)
        decoded = self.Decoder(encoded, hidden_state)
        return decoded


####  non definitivo

class Encoder_large(nn.Module):
    def __init__(self, sq_len, num_feat, exp_dim, compr_dim, num_layers, bottleneck_len=16):
        super().__init__()
        self.sq_len = sq_len
        self.bottleneck_len = bottleneck_len
        
        self.El1 = nn.LSTM(input_size=num_feat, hidden_size=exp_dim, num_layers=num_layers, batch_first=True)
        self.ln1 = nn.LayerNorm(exp_dim) 
        self.El2 = nn.LSTM(input_size=exp_dim, hidden_size=compr_dim, num_layers=num_layers, batch_first=True)
        self.ln2 = nn.LayerNorm(compr_dim)
        
        # Max Pooling to preserve wave peaks/amplitudes safely
        self.pool = nn.AdaptiveMaxPool1d(bottleneck_len)
           
    def forward(self, item):
        item, _ = self.El1(item)
        item = self.ln1(item)
        item, (h_n, c_n) = self.El2(item)
        item = self.ln2(item)
        
        # Transpose for spatial pooling: (Batch, Channels, Length)
        item = item.transpose(1, 2)           
        item = self.pool(item)                
        item = item.transpose(1, 2)           # Back to (Batch, Length, Channels)
        
        return item, (h_n, c_n)


class Decoder_large(nn.Module):
    def __init__(self, sq_len, num_feat, exp_dim, compr_dim, num_layers, bottleneck_len=16):
        super().__init__()
        self.sq_len = sq_len
        
        # The spontaneous, learnable time stretch
        self.time_stretch = nn.Linear(bottleneck_len, sq_len)
        
        self.Dl1 = nn.LSTM(input_size=compr_dim, hidden_size=compr_dim, num_layers=num_layers, batch_first=True)
        self.ln1 = nn.LayerNorm(compr_dim)
        self.Dl2 = nn.LSTM(input_size=compr_dim, hidden_size=exp_dim, num_layers=num_layers, batch_first=True)
        
        self.TimeDistributed = nn.Conv1d(exp_dim, num_feat, kernel_size=1)

    def forward(self, item, encoder_state):
        # 1. Let the network LEARN how to stretch 16 steps back to 389
        item = item.transpose(1, 2)           
        item = self.time_stretch(item)        
        item = item.transpose(1, 2)           
        
        # 2. Pass through LSTMs (using the Encoder's final state to jumpstart it!)
        item, _ = self.Dl1(item, encoder_state)
        item = self.ln1(item)
        item, _ = self.Dl2(item)
        
        # 3. Transpose for Conv1d, then transpose back to return
        item = item.transpose(1, 2)           
        item = self.TimeDistributed(item)     
        item = item.transpose(1, 2)           
        
        return item


class AE_multi_large(nn.Module):
    def __init__(self, sq_len, num_feat, exp_dim, compr_dim, num_layers, bottleneck_len=16):
        super().__init__()
        # Pass bottleneck_len into both Encoder and Decoder
        self.Encoder = Encoder_large(sq_len, num_feat, exp_dim, compr_dim, num_layers, bottleneck_len)
        self.Decoder = Decoder_large(sq_len, num_feat, exp_dim, compr_dim, num_layers, bottleneck_len)
        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'time_stretch' in name: # Initialize the new stretch layer
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)
            elif 'TimeDistributed' in name:
                if 'weight' in name:
                    nn.init.kaiming_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)
                    
    def forward(self, item):
        encoded, hidden_state = self.Encoder(item)
        decoded = self.Decoder(encoded, hidden_state)
        return decoded
    
