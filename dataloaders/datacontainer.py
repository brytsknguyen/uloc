import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dataloaders.params import INPUT_NAMES
from metrics import AverageMeter, Result

class ResultContainer:
    def __init__(self, y_target):
        self.trajectory_container = Trajectory(y_target)
        self.avg_meter = AverageMeter()
        self.result = Result()

    def accum(self, y_pred, y_gt):
        self.trajectory_container.accum(y_pred, y_gt)

    def get_result(self):
        return self.trajectory_container.get_results()


class Trajectory:
    def __init__(self, y_target):
        self.Y_target = y_target
        self.y_gt_set = None
        self.y_pred_set = None
        self.is_initial = True

    def accum(self, y_pred, y_gt):
        if self.is_initial:
            self.y_gt_set = y_gt
            self.y_pred_set = y_pred
            self.is_initial = False
        else:
            self.y_gt_set = np.concatenate((self.y_gt_set, y_gt), axis=0)
            self.y_pred_set = np.concatenate((self.y_pred_set, y_pred), axis=0)

    def get_results(self):
        return self.y_gt_set, self.y_pred_set

if __name__ == "__main__":
    pass
