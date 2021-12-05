import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

from .util import *
from .client import Client


def read_libsvm(name):
    x, y = load_svmlight_file(f'{DATA_PATH}/{name}.libsvm')
    x = x.toarray().astype(np.float32)
    y = y.astype('str')
    test_path = Path(f'{DATA_PATH}/{name}_test.libsvm')
    if test_path.exists() and name != 'a9a':
        xt, yt = load_svmlight_file(str(test_path))
        xt = xt.toarray().astype(np.float32)
        yt = yt.astype('str')
        return x, xt, y, yt
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        return x_train, x_test, y_train, y_test


def read_dataset(name):
    if name == 'ddos':
        _path = DATA_PATH / 'ddos_noniid'
        xs, ys = [], []
        files = list(_path.rglob('*.csv'))
        for fp in tqdm(files, desc='Reading'):
            data = pd.read_csv(str(fp), skipinitialspace=True, low_memory=False)
            data['SimillarHTTP'] = 0.
            x = data.iloc[:, -80:-1].to_numpy().astype(np.float32)
            y = data.iloc[:, -1].to_numpy().astype('str').tolist()
            x[x == np.inf] = 1.
            x[np.isnan(x)] = 0.
            xs += x.tolist()
            ys += y
        x_train, x_test, y_train, y_test = train_test_split(np.array(xs), np.array(ys), test_size=0.2)
        return x_train, x_test, y_train, y_test
    elif name in ['a9a', 'sen', 'sensit', 'covtype', 'HIGGS', 'cod-rna', 'mushrooms', 'mnist']:
        return read_libsvm(name)


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
    c.dataset = ys
    ret.append(c)
    return ret
