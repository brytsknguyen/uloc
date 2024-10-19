import numpy as np
import math
# Evaluation class

class Result(object):
    def __init__(self,half=0):
        self.abs_diff  = None
        self.N         = 0
        self.rmse      = 0.0
        self.mean      = 0.0
        self.median    = 0.0
        self.var       = 0.0
        self.error_max = 0.0
        self.half      = half

    def evaluate(self, output, target):
        if self.half:
            diff           = (output[:,:3]) - (target[:,:3])
        else:
            diff           = (output[:,:3]+output[:,3:])/2 - (target[:,:3]+target[:,3:])/2
        self.abs_diff  = np.abs(diff)
        self.N         = len(diff)
        err            = np.linalg.norm(self.abs_diff, axis=1).reshape(-1,1)
        self.rmse      = np.sqrt(err.transpose()@err/len(err))[0][0]

        self.rmse2     = math.sqrt(np.mean(np.power(self.abs_diff, 2)))
        self.var       = np.var(self.abs_diff)
        self.mean      = np.mean(self.abs_diff)
        self.median    = np.median(self.abs_diff)
        self.error_max = np.amax(self.abs_diff)

    def __add__(self, other):
        if isinstance(other, Result):
            if self.abs_diff is None:
                self.abs_diff  = other.abs_diff
                self.N         = other.N       
                self.rmse      = other.rmse    
                self.mean      = other.mean    
                self.median    = other.median  
                self.var       = other.var     
                self.error_max = other.error_max
            else:
                self.abs_diff  = np.concatenate((self.abs_diff, other.abs_diff))
                Nnew           = self.N + other.N
                err            = np.linalg.norm(self.abs_diff, axis=1).reshape(-1,1)
                self.rmse      = np.sqrt(err.transpose()@err/len(err))[0][0]
                self.rmse2      = math.sqrt(np.mean(np.power(self.abs_diff, 2)))
                self.N          = Nnew
                # self.rmse      = math.sqrt(np.mean(np.power(self.abs_diff, 2)))
                # self.N         = self.N + other.N
                self.var       = np.var(self.abs_diff)
                self.mean      = np.mean(self.abs_diff)
                self.median    = np.median(self.abs_diff)
                self.error_max = np.amax(self.abs_diff)
        return self