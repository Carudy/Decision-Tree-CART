import random
from collections import defaultdict

NP = 100007


class Client:
    def __init__(self, idx):
        self.id = idx
        self.keys = defaultdict(int)
        self.receive_keys = {}
        self.dataset = None
        self.attrs = set()

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

    def get_sample(self, j):
        vals = self.dataset.iloc[j, :].tolist()
        ret = {}
        for k, v in zip(self.attrs, vals):
            ret[k] = v
        return ret
