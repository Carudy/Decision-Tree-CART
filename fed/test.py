from sklearn.metrics import accuracy_score
from .encrypt import *
from model import Vfdt


def center_test(center, xs, ys, keys):
    _xs = [enc_dict({str(k): x[k] for k in range(len(x))}, keys) for x in xs]
    pred = center.tree.predict(_xs)
    acc = accuracy_score(pred, [hash_sha(str(y)) for y in ys])
    print(f'Encrypted tree acc: {acc * 100.}%')


def pure_test(x_train, x_test, y_train, y_test):
    pure_tree = Vfdt(attrs=[i for i in range(len(x_train[0]))], verbose=False)
    pure_tree.update(x_train, y_train)
    pred = pure_tree.predict(x_test)
    acc = accuracy_score(pred, y_test)
    print(f'Ori acc: {acc * 100.}%')
