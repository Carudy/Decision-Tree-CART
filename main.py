import time
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as SDT
from model import *
from encryption.oph import Pope
from read_data import *


def full_test(dataset, cmd=0b111, depths=[32]):
    print('Testing dataset: ' + dataset)
    X_train, X_test, y_train, y_test = read_dataset(dataset)
    print('Data read.')
    oph_a, oph_r = 1024, 4
    encoder = Pope(a=oph_a, r=oph_r)
    X_train_oph, X_test_oph = encoder.encode(X_train), encoder.encode(X_test)
    num_feature = len(X_test[0])
    info = ''
    print('Data preprocessed.')

    for max_depth in depths:
        info += str(max_depth) + '\n'
        print(max_depth)
        st = time.time()
        if cmd & 1:
            # Normal IDT
            tree = Vfdt(num_feature, delta=1e-7, nmin=10000, tau=0.5, max_depth=max_depth, regional=None)
            tree.partial_fit(X_train, y_train)
            y_pred = tree.predict(X_test)
            info_now = 'IDT Acc: {:.2f}'.format(accuracy_score(y_test, y_pred) * 100)
            info += info_now + ', '
            print(info_now, end=', ')
            ed = time.time()
            info_now = 'Cost {:.3f}s'.format(ed - st)
            print(info_now)
            info += info_now + '\n'
            st = ed
            del tree

        if cmd & 2:
            # OPH-IDT
            tree = Vfdt(num_feature, delta=1e-7, nmin=10000, tau=0.5, max_depth=max_depth, regional=None)
            tree.partial_fit(X_train_oph, y_train)
            y_pred = tree.predict(X_test_oph)
            info_now = 'OPH Acc: {:.2f}'.format(accuracy_score(y_test, y_pred) * 100)
            info += info_now + ', '
            print(info_now, end=', ')
            ed = time.time()
            info_now = 'Cost {:.3f}s'.format(ed - st)
            print(info_now)
            info += info_now + '\n'
            st = ed
            del tree

        if cmd & 4:
            # RC-IDT
            tree = Vfdt(num_feature, delta=1e-7, nmin=10000, tau=0.5, max_depth=max_depth, regional=oph_r)
            tree.partial_fit(X_train_oph, y_train)
            y_pred = tree.predict(X_test_oph)
            info_now = 'RC  Acc: {:.2f}'.format(accuracy_score(y_test, y_pred) * 100)
            info += info_now + ', '
            print(info_now, end=', ')
            ed = time.time()
            info_now = 'Cost {:.3f}s'.format(ed - st)
            print(info_now)
            info += info_now + '\n'
            st = ed
            del tree

        if cmd & 8:
            tree = SDT(max_depth=max_depth)
            tree.fit(X_train, y_train)
            y_pred = tree.predict(X_test)
            info_now = 'DT  Acc: {:.2f}'.format(accuracy_score(y_test, y_pred) * 100)
            info += info_now + ', '
            print(info_now, end=', ')
            ed = time.time()
            info_now = 'Cost {:.3f}s'.format(ed - st)
            print(info_now)
            info += info_now + '\n'
            del tree

        if cmd & 16:
            tree = SDT(max_depth=max_depth)
            tree.fit(X_train_oph, y_train)
            y_pred = tree.predict(X_test_oph)
            info_now = 'DTO Acc: {:.2f}'.format(accuracy_score(y_test, y_pred) * 100)
            info += info_now + ', '
            print(info_now, end=', ')
            ed = time.time()
            info_now = 'Cost {:.3f}s'.format(ed - st)
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
