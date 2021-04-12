import numpy as np
import random, os, time
# from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier as HT
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_svmlight_file 
from sklearn.tree import DecisionTreeClassifier as SDT

from model.vfdt import Vfdt
from model.efdt import Efdt
from oph import OPH_server
import pandas as pd
from torch import save, load

DATA_PATH = './data'

ATTACK_TYPES = {
    'snmp': 0,
    'portmap': 1,
    'syn': 2,
    'dns': 3,
    'ssdp': 4,
    'webddos': 5,
    'mssql': 6,
    'tftp': 7,
    'ntp': 8,
    'udplag': 9,
    'ldap': 10,
    'netbios': 11,
    'udp': 12,
    'benign': 13,
}

def read_libsvm(file):
    X, y = load_svmlight_file('{}/{}.libsvm'.format(DATA_PATH, file))
    X = X.toarray().astype(np.float32)
    return X, y

def read_csv(file):
    data = pd.read_csv('{}/{}.csv'.format(DATA_PATH, file), header=None, skipinitialspace=True, low_memory=False)
    X = data.iloc[:, :-1].to_numpy().astype(np.float32)
    Y = data.iloc[:, -1].to_numpy()
    return X, Y

def read_data(data_name, fmt):
    if fmt=='libsvm': 
        return read_libsvm(data_name)
    elif fmt=='csv':
        return read_csv(data_name)
    elif fmt=='bin':
        x, y = load('{}/{}'.format(DATA_PATH, data_name))
        x[x == np.inf] = 1.
        x[np.isnan(x)] = 0.
        # binarize
        # y = np.array([1 if t!=13 else 0 for t in y])
        return (x, y)

def has_test(data_name):
    for root, dirs, fnames in os.walk(DATA_PATH):
        for fname in fnames:
            if data_name + '_test' in fname: return True
    return False

def read_dataset(data_name):
    if data_name in ['a9a', 'covtype', 'mushrooms', 'HIGGS', 'sen', 'senit']:
        fmt = 'libsvm'
    elif data_name in ['covertype']:
        fmt = 'csv'
    else:
        fmt = 'bin'

    X, y = read_data(data_name, fmt)
    print('Date size: {}'.format(len(y)), X.shape if isinstance(X, np.ndarray) else type(X[0]))
    if has_test(data_name):
        print('Test dataset found.')
        X_test, y_test = read_data(data_name + '_test', fmt)
    else:
        X, X_test, y, y_test = train_test_split(X, y, test_size=0.25)
    rd = random.randint(0, 1024)
    random.Random(rd).shuffle(X)
    random.Random(rd).shuffle(y)
    return X, X_test, y, y_test

def full_test(dataset, cmd=0b111, depths=[32]):
    print('Testing dataset: ' + dataset)
    X_train, X_test, y_train, y_test = read_dataset(dataset)
    print('Data read.')
    # print(len(y_test), len([i for i in y_test if i==0]))
    oph_a, oph_r = 1024, 4
    encoder = OPH_server(a=oph_a, r=oph_r)
    X_train_oph, X_test_oph = encoder.encode(X_train), encoder.encode(X_test)
    num_feature = len(X_test[0])
    info = ''
    print('Data preprocessed.')

    for max_depth in depths:
        info += str(max_depth) + '\n'
        print(max_depth)
        st = time.time()
        if cmd&1:
            # Normal IDT
            tree = Vfdt(num_feature, delta=1e-7, nmin=10000, tau=0.5, max_depth=max_depth, regional=None)
            tree.partial_fit(X_train, y_train)
            y_pred = tree.predict(X_test)
            info_now = 'IDT Acc: {:.2f}'.format(accuracy_score(y_test, y_pred) * 100)
            info += info_now + ', '
            print(info_now, end=', ')
            ed = time.time()
            info_now = 'Cost {:.3f}s'.format(ed-st)
            print(info_now)
            info += info_now + '\n'
            st = ed
            del tree

        if cmd&2:
            # OPH IDT
            tree = Vfdt(num_feature, delta=1e-7, nmin=10000, tau=0.5, max_depth=max_depth, regional=None)
            tree.partial_fit(X_train_oph, y_train)
            y_pred = tree.predict(X_test_oph)
            info_now = 'OPH Acc: {:.2f}'.format(accuracy_score(y_test, y_pred) * 100)
            info += info_now + ', '
            print(info_now, end=', ')
            ed = time.time()
            info_now = 'Cost {:.3f}s'.format(ed-st)
            print(info_now)
            info += info_now + '\n'
            st = ed
            del tree

        if cmd&4:
            # OPH IDT with Regional Counting
            tree = Vfdt(num_feature, delta=1e-7, nmin=10000, tau=0.5, max_depth=max_depth, regional=oph_r)
            tree.partial_fit(X_train_oph, y_train)
            y_pred = tree.predict(X_test_oph)
            info_now = 'RC  Acc: {:.2f}'.format(accuracy_score(y_test, y_pred) * 100)
            info += info_now + ', '
            print(info_now, end=', ')
            ed = time.time()
            info_now = 'Cost {:.3f}s'.format(ed-st)
            print(info_now)
            info += info_now + '\n'
            st = ed
            del tree

        if cmd&8:
            tree = SDT(max_depth=max_depth)
            tree.fit(X_train, y_train)
            y_pred = tree.predict(X_test)
            info_now = 'DT  Acc: {:.2f}'.format(accuracy_score(y_test, y_pred) * 100)
            info += info_now + ', '
            print(info_now, end=', ')
            ed = time.time()
            info_now = 'Cost {:.3f}s'.format(ed-st)
            print(info_now)
            info += info_now + '\n'
            del tree

        if cmd&16:
            tree = SDT(max_depth=max_depth)
            tree.fit(X_train_oph, y_train)
            y_pred = tree.predict(X_test_oph)
            info_now = 'DTO Acc: {:.2f}'.format(accuracy_score(y_test, y_pred) * 100)
            info += info_now + ', '
            print(info_now, end=', ')
            ed = time.time()
            info_now = 'Cost {:.3f}s'.format(ed-st)
            print(info_now)
            info += info_now + '\n'
            del tree

    open('result.txt', 'w').write(info)

if __name__ == '__main__':
    full_test('sen', cmd=0b111, depths=[32])

# all_ddos
# sen 67.64 67.49 67.82
# sen 67.17 68.55 68.55
# HIGGS 61.88 61.92 61.92
# covtype 78.23 78.43
# bin DDOS 99.75 99.46 99.46
# DDOS 58.34 53.01 57.30
# DDOS 58.13 51.55 55.38

# a9a 