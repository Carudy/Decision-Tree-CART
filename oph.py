import numpy as np

# more complicated ope required

class OPH_server():
    def __init__(self, a=128, r=2):
        self.a = a
        self.r = r

    def encode(self, X):
        sign = (X>0).astype(np.float32) * 1. + (X<0).astype(np.float32) * -1.
        return (X * self.a) + (np.random.rand(X.shape[0], X.shape[1]).astype(np.float32) * sign * self.r)