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

    def enc_sample(self, i):
        ret = {
            'pid': self.id,
            'uid': hash_sha(self.dataset[i]['uid']),
        }
        for k, v in self.dataset[i].items():
            if k != 'uid':
                ret[hash_sha(k)] = ope(v, self.keys[k], dp=self.dp)
        return ret

    def split_data(self):
        _l = list(range(len(self.dataset)))
        if ARGS.shuffle:
            random.shuffle(_l)
        self.data_batches = np.array_split(_l, ARGS.n_round)
        self.to_send_id = 0

    def send_batch(self):
        if self.to_send_id >= len(self.data_batches):
            print(f'Party {self.id} has already sent all data.')
            return
        for i in self.data_batches[self.to_send_id]:
            self.center.receive_sample(
                self.enc_sample(i) if self.attrs != 'label' else {k: hash_sha(v) for k, v in self.dataset[i].items()})
        self.to_send_id += 1
