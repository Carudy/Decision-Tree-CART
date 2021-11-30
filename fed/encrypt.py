import hashlib
from functools import lru_cache

import numpy as np


@lru_cache
def hash_sha(x):
    return hashlib.sha256(x.encode()).hexdigest()


def ope(x, k, gamma=0.04, dp=True):
    if dp:
        return k * x + np.random.laplace(scale=(1. / gamma))
    else:
        return k * x


def enc_dict(x, keys):
    ret = {}
    for k, v in x.items():
        ret[hash_sha(k)] = ope(v, keys[k])
    return ret
