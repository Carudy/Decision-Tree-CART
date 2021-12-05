import copy
import time
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
    pure_tree = Vfdt(attrs=[i for i in range(len(x_train[0]))], verbose=True, nmin=ARGS.nmin, record=ARGS.record)
    st = time.time()
    pure_tree.update(x_train, y_train)
    tp = (time.time() - st) * 1000.
    pred = pure_tree.predict(x_test)
    acc = accuracy_score(pred, y_test)
    log(f'IDT acc: {acc * 100.:.3f}%')
    log(f'IDT cost time: {tp:.3f}ms')
    log(f'IDT #record: {pure_tree.root.num_vals()}')
    if ARGS.record:
        log(str(pure_tree.xp))
        log(str(pure_tree.yp['num_record']))
        log(str(pure_tree.yp['speed']))
        log('\n')


def ope_test(x_train, x_test, y_train, y_test, keys, ope=True, rc=True):
    xt = copy.deepcopy(x_train)
    xc = copy.deepcopy(x_test)
    mt = list(keys.values())
    for i in range(len(xt)):
        pt = [np.random.laplace(scale=(1. / ARGS.gamma)) for _ in range(len(mt))]
        xt[i] = xt[i] * mt + pt
    for i in range(len(xc)):
        pt = [np.random.laplace(scale=(1. / ARGS.gamma)) for _ in range(len(mt))]
        xc[i] = xc[i] * mt + pt
    if ope:
        ope_tree = Vfdt(attrs=[i for i in range(len(x_train[0]))], verbose=True, nmin=ARGS.nmin, record=ARGS.record)
        st = time.time()
        ope_tree.update(xt, y_train)
        tp = (time.time() - st) * 1000.
        pred = ope_tree.predict(xc)
        acc = accuracy_score(pred, y_test)
        log(f'OPE-IDT acc: {acc * 100.:.3f}%')
        log(f'OPE-IDT cost time: {tp:.3f}ms')
        log(f'OPE-IDT #record: {ope_tree.root.num_vals()}')
        if ARGS.record:
            log(str(ope_tree.yp['num_record']))
            log(str(ope_tree.yp['speed']))
            log('\n')

    if rc:
        rc_tree = Vfdt(attrs=[i for i in range(len(x_train[0]))], verbose=True, nmin=ARGS.nmin, record=ARGS.record,
                       regional_count=(ARGS.zeta / ARGS.gamma))
        st = time.time()
        rc_tree.update(xt, y_train)
        tp = (time.time() - st) * 1000.
        pred = rc_tree.predict(xc)
        acc = accuracy_score(pred, y_test)
        log(f'RC-IDT acc: {acc * 100.:.3f}%')
        log(f'RC-IDT cost time: {tp:.3f}ms')
        log(f'RC-IDT #record: {rc_tree.root.num_vals()}')
        if ARGS.record:
            log(str(rc_tree.yp['num_record']))
            log(str(rc_tree.yp['speed']))
