import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class eca_layer(nn.Module):

    def __init__(self,k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class rfbModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, rnn_type="BiLSTM"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.eca = eca_layer(1)
        self.relu = torch.nn.ReLU(inplace=False)
        
        if rnn_type == "BiLSTM":
            self.rnn = torch.nn.LSTM(input_dim, hidden_dim, bidirectional=True, num_layers=1, batch_first=True)
        elif rnn_type == "LSTM":
            self.rnn = torch.nn.LSTM(input_dim, hidden_dim, bidirectional=False, num_layers=1, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = torch.nn.GRU(input_dim, hidden_dim, bidirectional=False, num_layers=1, batch_first=True)
        elif rnn_type == "BiGRU":
            self.rnn = torch.nn.GRU(input_dim, hidden_dim, bidirectional=True, num_layers=1, batch_first=True)
        elif rnn_type == "RNN":
            self.rnn = torch.nn.RNN(input_dim, hidden_dim, bidirectional=False, num_layers=1, batch_first=True)
        elif rnn_type == "BiRNN":
            self.rnn = torch.nn.RNN(input_dim, hidden_dim, bidirectional=True, num_layers=1, batch_first=True)
        else:
            raise ValueError(f"Unsupported model type: {rnn_type}")

        self.fc = torch.nn.Linear(hidden_dim * (2 if "Bi" in rnn_type else 1), output_dim, bias=True)
        self.bn = torch.nn.BatchNorm1d(output_dim, affine=False)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.relu(x)
        x_p = x  

        x = x.transpose(1, 2)
        x = self.eca(x)
        x = x.transpose(1, 2)
        x = x + x_p  

        bs, seq, hs = x.size()
        x = x.reshape(bs * seq, hs)
        x = self.fc(x)
        x = self.bn(x)
        x = x.view(bs, seq, self.output_dim)

        return x
    
class MyRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, coding_dim, depth=3,rnn_type="BiLSTM"):
        super().__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth
        self.rfb = torch.nn.Sequential()

        for level in range(0, depth):
            chs = int(hidden_dim / 2**(level))
            nhs = int(hidden_dim / 2**(level+1))
            if level == 0:
                self.rfb.append(rfbModel(coding_dim, chs, nhs,rnn_type=rnn_type))
            else:
                self.rfb.append(rfbModel(chs, chs, nhs,rnn_type=rnn_type))

        self.relu  = torch.nn.ReLU(inplace=False)
        lhs = int(self.hidden_dim / 2**(depth))
        self.fcend = torch.nn.Linear(lhs, output_dim, bias=True)    
        self.coding = torch.nn.Linear(input_dim,coding_dim, bias=True)   
        
    def forward(self, x):
        bs, seq_len, input_dim = x.size()
        x = self.relu(self.coding(x))
        x = self.rfb(x)
        bs, seq, hs = x.size()
        x = x.reshape(bs * seq, hs)
        x = self.relu(x)
        x = self.fcend(x)
        x = x.view(bs, seq, self.output_dim)

        return x

if __name__ == "__main__":
    seqlen      = 20
    xdim        = 20
    ydim        = 6
    coding_dim  = 128
    hiddendim   = 256
    model = MyRNN(input_dim=xdim, hidden_dim=hiddendim, output_dim=ydim,coding_dim=coding_dim,rnn_type="GRU")
    model_parameters = list(model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Num. of parameters: ", params)
    x = torch.randn(32, seqlen, xdim)  # Example input tensor
    output = model(x)
    print(output.shape)

