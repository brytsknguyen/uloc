import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# Make the dataloader
class MCDLoader(Dataset):
    def __init__(self, datapath, seqlen, Xscaler, Yscaler,min_anc,half=0):

        self.datapath = datapath
        self.seqlen = seqlen

        # Length of the sequence
        self.num_samples = 0

        # Create scaler for data
        self.Xscaler = Xscaler
        self.Yscaler = Yscaler
        self.half    = half
        X = []
        Y = []

        # Store the data in each csv file
        data = []

        # Load the csv in data path
        csv_files = set()
        try:
            csv_files = set([datapath + '/' + file for file in os.listdir(datapath)])
        except:
            print('Path is not dir')
            if '.csv' in datapath:
                csv_files = [datapath]

        csv_files = sorted(csv_files)

        print(csv_files)

        for file in csv_files:
            
            # print('Loading', file)
            
            data = np.loadtxt(file, delimiter=',')

            t_ = data[:, 0]
            X_ = data[:, 7::3]
            Y_ = data[:, [1,2,3,4,5,6]]
            if self.half:
                X_ = X_[:,10:]
                Y_ = Y_[:,3:]
                # X_ = X_[:,:10]
                # Y_ = Y_[:,:3]
            # Y_ = (Y_[:,:2]+Y_[:,2:])/2
            # Rescale and sample the sequence in each chunk
            X_ = self.Xscaler.transform(X_)
            Y_ = self.Yscaler.transform(Y_)

            data_len = (X_.shape[0] - seqlen) + 1

            # Create the sequences from this chunk
            for idx in range(0, data_len):

                tseq  = t_[idx:idx+seqlen]
                dtseq = np.fabs(max(tseq) - min(tseq))

                # Check if the number of anchors available is 
                Xseq = X_[idx:idx+seqlen, :]
                Yseq = Y_[idx:idx+seqlen, :]

                have_enough_anc = False

                for seqIdx, x in enumerate(Xseq):
                    # print('idx', idx, seqIdx, np.count_nonzero(x[0:10]), x)
                    if np.count_nonzero(x[0:10]) >= min_anc or np.count_nonzero(x[10:]) >= min_anc:
                        have_enough_anc = True

                if dtseq < seqlen*0.05*2 and have_enough_anc:
                    X.append(Xseq)
                    Y.append(Yseq)

            print(f'Loading file {file}. Size: {data[-1].shape[0]}')
            
        # print(f'Storing input-output sequences')        
        # Convert data to torch
        self.X = torch.from_numpy(np.array(X))
        self.Y = torch.from_numpy(np.array(Y))

        # Size
        self.num_samples = self.X.shape[0]
        print(f'Number of samples: {self.num_samples}')

    def getX(self):
        # return torch.concatenate((self.X[0, :, :], self.X[:, -1, :]))
        return self.X
        
    def getY(self):
        # return torch.concatenate((self.X[0, :, :], self.X[1:, -1, :]))
        return self.Y

    def __getitem__(self, index):
        
        return self.X[index, :, :], self.Y[index, :, :]

    def __len__(self):
        return self.num_samples
    

def obtain_scale(train_path,all_path,half=0):
    Xscaler = MinMaxScaler()
    Yscaler = MinMaxScaler()

    csv_files = os.listdir(train_path) #+ os.listdir(test_path)
    for file in csv_files:
        
        data = np.loadtxt(all_path + '/' + file, delimiter=',')
        X_ = data[:, 7::3]
        Y_ = data[:, [1,2,3,4,5,6]]
        # Y_ = (Y_[:,:2]+Y_[:,2:])/2
        if half:
            X_ = X_[:,:10]
            Y_ = Y_[:,:3]
        # Update the scale with the data
        Xscaler.partial_fit(X_)
        Yscaler.partial_fit(Y_)
    
    return Xscaler,Yscaler