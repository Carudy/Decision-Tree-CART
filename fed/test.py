import copy

from sklearn.metrics import accuracy_score
from .util import *
from .encrypt import *
from model import Vfdt


def center_test(center, xs, ys, keys):
    _xs = [enc_dict({str(k): x[k] for k in range(len(x))}, keys) for x in xs]
    pred = center.tree.predict(_xs)
    acc = accuracy_score(pred, [hash_sha(str(y)) for y in ys])
    log(f'Encrypted tree acc: {acc * 100.}%')


def pure_test(x_train, x_test, y_train, y_test):
    pure_tree = Vfdt(attrs=[i for i in range(len(x_train[0]))], verbose=True)
    pure_tree.update(x_train, y_train)
    pred = pure_tree.predict(x_test)
    acc = accuracy_score(pred, y_test)
    log(f'IDT acc: {acc * 100.}%')


def ope_test(x_train, x_test, y_train, y_test, keys):
    xt = copy.deepcopy(x_train)
    xc = copy.deepcopy(x_test)
    mt = list(keys.values())
    for i in range(len(xt)):
        pt = [np.random.laplace(scale=(1. / ARGS.gamma)) for _ in range(len(mt))]
        xt[i] = xt[i] * mt + pt
    for i in range(len(xc)):
        pt = [np.random.laplace(scale=(1. / ARGS.gamma)) for _ in range(len(mt))]
        xc[i] = xc[i] * mt + pt
    ope_tree = Vfdt(attrs=[i for i in range(len(x_train[0]))], verbose=True, nmin=ARGS.nmin)
    ope_tree.update(xt, y_train)
    pred = ope_tree.predict(xc)
    acc = accuracy_score(pred, y_test)
    log(f'OPE-IDT acc: {acc * 100.}%')

    rc_tree = Vfdt(attrs=[i for i in range(len(x_train[0]))], verbose=True, nmin=ARGS.nmin,
                   regional_count=(1. / ARGS.gamma))
    rc_tree.update(xt, y_train)
    pred = ope_tree.predict(xc)
    acc = accuracy_score(pred, y_test)
    log(f'RC-IDT acc: {acc * 100.}%')
