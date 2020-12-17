import numpy as np
import random, os
# from skmultiflow.trees import ExtremelyFastDecisionTreeClassifier as XDT
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_svmlight_file 

from model.vfdt import Vfdt
from oph import OPH_server

DATA_PATH = './data'

def read_libsvm(file):
    X, y = load_svmlight_file('{}/{}.libsvm'.format(DATA_PATH, file))
    X = X.toarray().astype(np.float32)
    return X, y

def read_data(data_name, fmt):
    if fmt=='libsvm': return read_libsvm(data_name)

def has_test(data_name):
    for root, dirs, fnames in os.walk(DATA_PATH):
        for fname in fnames:
            if data_name + '_test' in fname: return True
    return False

def read_dataset(data_name, fmt='libsvm'):
    X, y = read_data(data_name, fmt)
    print('Date size: {}'.format(len(y)), X.shape if isinstance(X, np.ndarray) else type(X[0]))
    if has_test(data_name):
        print('Test dataset found.')
        X_test, y_test = read_data(data_name + '_test', fmt)
    else:
        X, X_test, y, y_test = train_test_split(X, y, test_size=0.33)
    print('Data preprocessed.')
    return X, X_test, y, y_test

if __name__ == '__main__':
    _OPH = False
    if _OPH: print("OPH enabled.")
    X_train, X_test, y_train, y_test = read_dataset('covtype', fmt='libsvm')
    # exit()
    
    encoder = OPH_server(a=32, r=20)
    tree = Vfdt(len(X_test[0]))
    
    if _OPH:
        print(X_test[0][:5])
        X_train, X_test = encoder.encode(X_train), encoder.encode(X_test)
        print(X_test[0][:5])
    
    tree.update(X_train, y_train)
    
    y_pred = tree.predict(X_test)
    print('Acc: {:.2f}'.format(accuracy_score(y_test, y_pred) * 100), end=' ')
# HIGGS     70.37   62.87
# real-sim 86.64  
# covtype 78.23  78.43