import time
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as SDT
from model import *
from encryption.oph import Pope
from read_data import *


def test_model(tree, X_train, Y_train, X_test, Y_test):
    st = time.time()
    tree.update(X_train, Y_train)
    y_pred = tree.predict(X_test)
    acc = accuracy_score(Y_test, y_pred) * 100
    ed = time.time()
    timep = ed - st
    return {
        'time': timep,
        'acc': acc,
    }


def test_models(dataset, models):
    print('Testing dataset: ' + dataset)
    X_train, X_test, Y_train, Y_test = read_dataset(dataset)
    print('Data read.')
    oph_a, oph_r = 512, 16
    encoder = Pope(a=oph_a, r=oph_r)
    X_train_oph, X_test_oph = encoder.encode(X_train), encoder.encode(X_test)
    num_feature = len(X_test[0])
    print('Data preprocessed.')

    info = ''
    for name in models:
        if name == 'IDT':
            # Normal IDT
            tree = Vfdt(num_feature)
            res = test_model(tree, X_train, Y_train, X_test, Y_test)
        elif name == 'OPH':
            tree = Vfdt(num_feature)
            res = test_model(tree, X_train_oph, Y_train, X_test_oph, Y_test)
        elif name == 'RC':
            tree = Vfdt(num_feature, regional_count=oph_r)
            res = test_model(tree, X_train_oph, Y_train, X_test_oph, Y_test)
        info += f'{name}: ACC:{res["acc"]:.2f} time:{res["time"]:.2f}\n'

    print(info)
    open('result.txt', 'w').write(info)


if __name__ == '__main__':
    for _ in range(3):
        test_models('covtype', models=('IDT', 'OPH', 'RC'))
