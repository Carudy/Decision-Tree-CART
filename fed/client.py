import random
from collections import defaultdict
from .encrypt import *

NP = 100007


class Client:
    def __init__(self, idx):
        self.id = idx
        self.keys = defaultdict(int)
        self.receive_keys = {}
        self.dataset = None
        self.attrs = set()
        self.data_batches = None
        self.to_send_id = 0
        self.center = None
        self.dp = True

    def send_keys(self, all_clients):
        for attr in self.attrs:
            _k = random.randint(0, NP - 1)
            self.keys[attr] = _k
        for client in all_clients:
            if client.id != self.id:
                client.receive_keys[self.id] = self.keys.copy()

    def calc_keys(self):
        for attr in self.attrs:
            for keys in self.receive_keys.values():
                self.keys[attr] = (self.keys[attr] + keys[attr]) % NP

    def get_sample(self, i):
        if isinstance(self.dataset, list):
            vals = self.dataset[i]
        else:
            vals = self.dataset.iloc[i, :].tolist()
        ret = {}
        for k, v in zip(self.attrs, vals):
            ret[k] = v
        return ret

    def enc_pure(self, x):
        ret = {}
        for k, v in x.items():
            ret[hash_sha(k)] = ope(v, self.keys[k], dp=self.dp)
        return ret

    def enc_sample(self, i):
        ret = {
            'pid': self.id,
            'uid': hash_sha(str(i)),
        }
        ret.update(self.enc_pure(self.get_sample(i)))
        return ret

    def split_data(self, n, shuffle=False):
        _l = list(range(len(self.dataset)))
        if shuffle:
            random.shuffle(_l)
        self.data_batches = np.array_split(_l, n)
        self.to_send_id = 0

    def send_batch(self):
        if self.to_send_id >= len(self.data_batches):
            print(f'Party {self.id} has already sent all data.')
            return
        for i in self.data_batches[self.to_send_id]:
            self.center.receive_sample(self.enc_sample(i) if self.attrs != 'label' else
                                       {'uid': hash_sha(str(i)), 'label': hash_sha(str(self.dataset[i]))})
        self.to_send_id += 1
