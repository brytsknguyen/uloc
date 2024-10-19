import os
import pandas as pd
import os.path
import numpy as np
import torch.utils.data as data
import dataloaders.transforms as transforms
from sklearn.preprocessing import MinMaxScaler
from dataloaders.params import INPUT_NAMES, X_KEY
SAFETY_FACTOR = 10
GT_LENGTH = 2
def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()

    csvs = [np.loadtxt(os.path.join(dir, csvname), delimiter=',') for csvname in classes]
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return csvs, class_to_idx


def make_dataset(csvs, seq_len, stride, interval):
    '''
    This function is necessary since RNNs takes input whose shape is [seq_len, x_dim]
    :return: parsed train/val data
    '''
    inputs = []
    window_size = 1 + (seq_len - 1) * interval

    for order, csv in enumerate(csvs):
        total_length = len(csv)
        num_idxes = int((total_length - window_size + 1)//stride)

        assert (num_idxes - 1) * stride + window_size - 1 < total_length

        for i in range(num_idxes):
            start_idx = i * stride
            item = (order, start_idx)
            inputs.append(item)

    return inputs


to_tensor = transforms.ToTensor()

class MyDataloader(data.Dataset):
    def __init__(self, root, type, scaler, Y_target, seq_len=128, stride=1, interval=1):
        """
        :param root:
        :param type:
        :param scaler:
        :param X_columns:
        :param Y_type:
        :param seq_len:
        :param stride: Interval btw time t-1 data and time t data
        :param interval: Interval in the input
        """
        csvs, class_to_idx = find_classes(root)
        self.csvs_raw = csvs
        self.class_to_idx = class_to_idx
        self.scaler = scaler
        self.type = type # train or val

        self.Y_target = Y_target
        self.seq_len = seq_len
        self.stride = stride  # Stride for window
        self.interval = interval  # Interval size btw each data in a window
        self.csvs_scaled = self.scale_inputs()

        self.inputs = make_dataset(self.csvs_scaled, seq_len=seq_len, stride=stride, interval=interval)
        print("Total ", len(self.inputs), " data are generated")

    def scale_inputs(self):
        csvs_scaled = []
        for csv_data in self.csvs_raw:
            X = csv_data[:, :-GT_LENGTH]
            Y = csv_data[:, -GT_LENGTH:]
            X_scaled = self.scaler.X_scaler.transform(X)
            Y_scaled = self.scaler.Y_scaler.transform(Y)
            data_scaled = np.concatenate((X_scaled, Y_scaled), axis=1)
            csvs_scaled.append(data_scaled)
        return csvs_scaled

    def get_input(self, id, idx):
        target_csv = self.csvs_scaled[id]
        # Note that stride is already considered in the function "make_dataset(~~)"
        target_idxes = [idx + self.interval * i for i in range(self.seq_len)]

        x = target_csv[target_idxes, :-GT_LENGTH]
        y = None
        if self.Y_target == "all":
            y = target_csv[target_idxes, -GT_LENGTH:]
        elif self.Y_target == "end":
            # maybe not in use
            y = target_csv[target_idxes[-1], -GT_LENGTH:]
        return x, y

    def __getraw__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (x, y) the transformed data.
        """
        csv_id, start_idx = self.inputs[index]
        x, y = self.get_input(csv_id, start_idx)
        return x, y, csv_id

    def __getitem__(self, index):

        x, y, csv_idx = self.__getraw__(index)

        tensor_x = to_tensor(x)
        tensor_y = to_tensor(y)

        if self.Y_target == "end":
            tensor_y = tensor_y.view(-1)

        return tensor_x, tensor_y

    def __len__(self):
        return len(self.inputs)


