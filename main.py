import argparse

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from fed import *


class MyParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--dataset', default='mushrooms')
        self.parser.add_argument('--n_client', default=5)
        self.parser.add_argument('--n_round', default=10)
        self.args = self.parser.parse_args()

    def __getitem__(self, item):
        return self.args.__getattribute__(item)

    def __getattr__(self, item):
        return self.args.__getattribute__(item)


def center_test(xs, ys, keys):
    def test_enc(x, keys):
        ret = {}
        for k, v in x.items():
            ret[hash_sha(k)] = ope(v, keys[k])
        return ret

    _xs = [test_enc({str(k): x[k] for k in range(len(x))}, keys) for x in xs]
    pred = center.tree.predict(_xs)
    acc = accuracy_score(pred, [hash_sha(str(y)) for y in ys])
    print(acc)


if __name__ == '__main__':
    args = MyParser()

    xs, ys = read_libsvm(args.dataset)
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2)
    attrs = [hash_sha(str(i)) for i in range(len(xs[0]))]
    clients = get_clients_with_xy(x_train, y_train, args.n_client)
    center = Center(attrs=attrs)

    for c in clients:
        c.center = center
        c.split_data(args.n_round)
        c.send_keys(clients)

    enc_keys = {str(i): 0 for i in range(len(attrs))}
    for c in clients:
        c.calc_keys()
        for i in range(len(attrs)):
            if enc_keys[str(i)] == 0 and c.keys[str(i)] != 0:
                enc_keys[str(i)] = c.keys[str(i)]

    for e in tqdm(range(args.n_round)):
        for c in clients:
            c.send_batch()
        center.aggregate()
        center.train()

    n_attrs = len(attrs)

    center_test(x_test, y_test, enc_keys)
    center.decode_tree(enc_keys)
    pred = center.tree.predict(x_test)
    acc = accuracy_score(pred, [hash_sha(str(y)) for y in y_test])
    print(acc)
