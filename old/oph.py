import numpy as np


class Pope:
    def __init__(self, a=512, r=2):
        self.a = a
        self.r = r

    def encode(self, X):
        sign = (X > 0).astype(np.int) * 1 + (X < 0).astype(np.int) * -1
        C = np.random.rand(*X.shape).astype(np.float32)
        D = 1 - np.power(2, -np.abs(X))
        return (X * self.a) + (C + D) * sign * self.r * 0.5


if __name__ == '__main__':
    A = np.array([[0.962, 0.63, 4.4]])
    e = Pope(256, 128)
    print(A)
    print(e.encode(A))
