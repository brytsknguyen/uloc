import torch
import torch.nn as nn
from util.eval import Result
import matplotlib.pyplot as plt
from IPython import display
from .mcd_uwbloader import MCDLoader,obtain_scale
from .Scaler import DataScaler
from .params import INPUT_NAMES, OUTPUT_NAMES, INPUT_LEN
from .uwb_dataloader import ronet_UWBDataloader
from torch.utils.data import  DataLoader
import os

def load_data(seqlen,batchsize,min_anc,training_dataset,train_on_slam,train_one_anchor,dataset_path):

    if training_dataset=='MCD':
        if train_on_slam:
            all_path    = os.path.join(dataset_path,'MCDUWB_slamprior/all')
            train_path  = os.path.join(dataset_path,'MCDUWB_slamprior/train')
            test_path   = os.path.join(dataset_path,'MCDUWB_slamprior/test')
        else:
            all_path    = os.path.join(dataset_path,'MCDUWB_gndtruth/all')
            train_path  = os.path.join(dataset_path,'MCDUWB_gndtruth/train')
            test_path   = os.path.join(dataset_path,'MCDUWB_gndtruth/test')

        Xscaler,Yscaler  = obtain_scale(train_path,all_path,half=train_one_anchor)
        train_dataset    = MCDLoader(train_path, seqlen=seqlen, Xscaler=Xscaler, Yscaler=Yscaler,min_anc=min_anc,half=train_one_anchor)
        test_dataset     = MCDLoader(test_path,  seqlen=seqlen, Xscaler=Xscaler, Yscaler=Yscaler,min_anc=min_anc,half=train_one_anchor)
        train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True,  num_workers=4, pin_memory=True)
        test_dataloader  = DataLoader(test_dataset , batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)

        return train_dataset ,test_dataset,train_dataloader,test_dataloader, Xscaler,Yscaler

    elif training_dataset=='RONET':

        scaledir = os.path.join(dataset_path, 'uwb_dataset/all')
        traindir = os.path.join(dataset_path, 'uwb_dataset/train')
        valdir = os.path.join(dataset_path, 'uwb_dataset/val')
        testdir = os.path.join(dataset_path, 'uwb_dataset/test')
        NUM_VAL_CSVS = len(os.listdir(valdir))
        mm_scaler = DataScaler(scaledir)
        train_dataset = ronet_UWBDataloader(traindir, 'train', mm_scaler, "all", seq_len=seqlen,
                                                        stride=1, interval=1)
        
        val_dataset = ronet_UWBDataloader(valdir, 'val', mm_scaler, "all", seq_len=seqlen,
                                    stride=1, interval=1)

        test_dataset = ronet_UWBDataloader(testdir, 'test', mm_scaler, "all", seq_len=seqlen,
                            stride=1, interval=1)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True,  num_workers=4, pin_memory=True)
        val_dataloader  = DataLoader(val_dataset , batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)
        test_dataloader  = DataLoader(test_dataset , batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)
        return train_dataloader,val_dataloader,test_dataloader,mm_scaler

    else:
        raise ValueError(f"Unsupported dataset type: {train_dataset}")

