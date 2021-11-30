import hashlib
import numpy as np


def hash_sha(x):
    return hashlib.sha256(x.encode()).hexdigest()


def ope(x, k, gamma=25):
    return k * x + np.random.laplace(scale=gamma)
