import torch
import torch.nn as nn
from util.eval import Result
import matplotlib.pyplot as plt
from IPython import display
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import  DataLoader
from models.mambaNet import MambaNet
from models.rnn_attention import MyRNN
import numpy as np

def load_model(model_name,input_dim,output_dim,coding_dim,num_layers, hidden_dim, device):
    if model_name=='MAMBA':
        model = MambaNet(input_dim,output_dim,coding_dim,num_layers,device)
    else:
        model = MyRNN(input_dim, hidden_dim, output_dim,coding_dim,rnn_type=model_name)

    model_parameters = list(model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Num. of parameters: ", params)
    
    return model