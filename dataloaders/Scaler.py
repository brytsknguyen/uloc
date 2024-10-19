import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataScaler:
    def __init__(self, root):
        self.X_scaler = MinMaxScaler()
        self.Y_scaler = MinMaxScaler()
        self.fit(root)

    def fit(self, root):
        for (path, _, files) in os.walk(root):
            for filename in files:
                try:
                    abs_path = os.path.join(path, filename)
                    if os.path.exists(abs_path):
                        print("Scaling ", abs_path, "...")
                        data = np.loadtxt(abs_path, delimiter=',')
                        X = data[:, :-2]
                        Y = data[:, -2:]
                        # For debuging
                        # print(X.shape, Y.shape)
                        self.X_scaler.partial_fit(X)
                        self.Y_scaler.partial_fit(Y)
                except KeyError:
                    print("May be a file is open!")

        print("Fitting for preprocessing of X complete. min :", self.X_scaler.data_min_, "max : ", self.X_scaler.data_max_)
        print("Fitting for preprocessing of Y complete. min :", self.Y_scaler.data_min_, "max : ", self.Y_scaler.data_max_)

    def undo_scale(self, Y):
        Y_undo_scaled = self.Y_scaler.inverse_transform(Y)
        return Y_undo_scaled

if __name__ == "__main__":
    scaler = DataScaler("/home/shapelim/ws/kari-lstm/uwb_dataset/all")
