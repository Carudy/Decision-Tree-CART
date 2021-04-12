import numpy as np

# more complicated & reasonable ope required

class OPH_server():
    def __init__(self, a=512, r=2):
        self.a = a
        self.r = r

    def encode(self, X):
        sign = (X>0).astype(np.int) * 1 + (X<0).astype(np.int) * -1
        C = np.random.rand(X.shape[0], X.shape[1]).astype(np.float32)
        D = 1 - np.power(1.001, -np.abs(X))
        return (X * self.a) + (C * 0.05 +  D * 0.95) * sign * self.r


if __name__ == '__main__':
	A = np.array([[0, 1], [-1, 0]])
	e = OPH_server(1024, 8)
	print(A)
	print(e.encode(A))