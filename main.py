from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from fed import *

if __name__ == '__main__':
    xs, ys = read_libsvm(ARGS.dataset)
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2)

    # test pure model
    pure_test(x_train, x_test, y_train, y_test)

    # init center & participants
    attrs = [hash_sha(str(i)) for i in range(len(xs[0]))]
    clients = get_clients_with_xy(x_train, y_train, ARGS.n_client)
    center = Center(attrs=attrs)

    # negotiate keys
    for c in clients:
        c.center = center
        c.split_data(ARGS.n_round)
        c.send_keys(clients)

    # get global keys for eval
    enc_keys = {str(i): 0 for i in range(len(attrs))}
    for c in clients:
        c.calc_keys()
        for i in range(len(attrs)):
            if enc_keys[str(i)] == 0 and c.keys[str(i)] != 0:
                enc_keys[str(i)] = c.keys[str(i)]

    # vertical FL training
    for e in tqdm(range(ARGS.n_round), desc='Train'):
        for c in clients:
            c.send_batch()
        center.aggregate()
        center.train()

    # test encrypted model
    center_test(center, x_test, y_test, enc_keys)

    # test decrypted model
    center.decode_tree(enc_keys)
    pred = center.tree.predict(x_test)
    acc = accuracy_score(pred, [hash_sha(str(y)) for y in y_test])
    print(f'Decrypted tree acc: {acc * 100.}%')
