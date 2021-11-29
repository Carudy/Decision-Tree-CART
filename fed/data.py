from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file

from .client import Client

DATA_PATH = Path(r'D:\work\py\Decision-Tree-CART\data')


def read_libsvm(name):
    x, y = load_svmlight_file(f'{DATA_PATH}/{name}.libsvm')
    x = x.toarray().astype(np.float32)
    y = y.astype('int')
    return x, y


def split_data(xs, n_type):
    n_attrs = len(xs[0])
    ds = pd.DataFrame(xs)
    attr_pieces = np.array_split(range(n_attrs), n_type)
    ret = []
    for piece in attr_pieces:
        data_piece = ds.iloc[:, piece[0]:piece[-1] + 1]
        ret.append(data_piece)
    return ret


def get_clients_with_xy(xs, ys, n_type):
    data_pieces = split_data(xs, n_type)
    ret = []
    n = 0
    for piece in data_pieces:
        c = Client(n)
        n += 1
        c.dataset = piece
        c.attrs = [str(i) for i in piece.columns]
        ret.append(c)
    c = Client(n)
    c.attrs = 'label'
    c.dataset = ys.astype('int').tolist()
    ret.append(c)
    return ret
