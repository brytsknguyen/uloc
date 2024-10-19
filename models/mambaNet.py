import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from mambapy.mamba import Mamba, MambaConfig
import numpy as np


class MambaNet(nn.Module):
    def __init__(self, input_dim,output_dim,coding_dim,num_layers,device):
        super().__init__()

        config = MambaConfig(d_model=coding_dim, n_layers=num_layers)
        mamba_extractor = Mamba(config)
        self.mamba  = mamba_extractor
        self.device  = device
        self.relu       = torch.nn.ReLU(inplace=False)
        self.fcend  = torch.nn.Linear(coding_dim, output_dim, bias=True)
        self.bn  = torch.nn.BatchNorm1d(output_dim, affine=False)
        self.coding = torch.nn.Linear(input_dim,coding_dim, bias=True)
        self.output_dim = output_dim   
        self.position_embeddings = nn.Embedding(100, coding_dim)
        self.position_embeddings.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):

        bs, seq_len, input_dim = x.size()
        x = self.relu(self.coding(x))
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(bs, seq_len)
        position_embeddings = self.position_embeddings(position_ids)
        x = x + position_embeddings
        x = self.mamba(x)
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
    device = "cuda"
    model = MambaNet(xdim,ydim,coding_dim,4,device).to(device)
    model_parameters = list(model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Num. of parameters: ", params)
    x = torch.randn(32, seqlen, xdim).to(device)  # Example input tensor
    output = model(x)
    print(output.shape)

