from collections import defaultdict

from model import Vfdt
from .encrypt import *


def pure_x(x):
    return {k: v for k, v in x.items() if k not in ['uid', 'pid', 'label']}


class Center:
    def __init__(self, attrs):
        self.attrs = attrs
        self.n_attr = len(attrs)
        self.tree = Vfdt(attrs=self.attrs, verbose=False, nmin=ARGS.nmin, record=ARGS.record,
                         regional_count=(ARGS.zeta / ARGS.gamma))
        self.new_data = defaultdict(list)
        self.samples = []
        self.train_x = []
        self.train_y = []
        self.labels = {}

    def receive_sample(self, x):
        if 'label' not in x:
            self.new_data[x['uid']].append(x)
        else:
            self.labels[x['uid']] = x['label']

    def dfs_sample(self, samples, s, now, attr_list, attr_id):
        if attr_id >= len(attr_list):
            samples.append(now.copy())
            return
        for v in s[attr_list[attr_id]]:
            now[attr_list[attr_id]] = v
            self.dfs_sample(samples, s, now, attr_list, attr_id + 1)

    def aggregate(self):
        for uid, xs in self.new_data.items():
            agg = defaultdict(list)
            for x in xs:
                for k, v in x.items():
                    if k not in ['pid', 'uid']:
                        agg[k].append(v)
            if len(set(agg.keys())) == self.n_attr:
                new_samples = []
                self.dfs_sample(new_samples, agg, {'uid': uid}, list(agg.keys()), 0)
                self.samples += new_samples

        # for uid in self.new_data:
        #     self.new_data[uid].clear()

        remain = []
        self.train_x = []
        self.train_y = []
        for x in self.samples:
            if x['uid'] in self.labels:
                self.train_x.append(pure_x(x))
                self.train_y.append(self.labels[x['uid']])
            else:
                remain.append(x)
        self.samples = remain[:]

    def train(self):
        self.tree.update(self.train_x, self.train_y)

    def decode_node(self, node, keys):
        if node.is_leaf():
            return
        for i in range(self.n_attr):
            if hash_sha(str(i)) == node.split_feature:
                node.split_feature = i
                node.split_value /= keys[str(i)]
                break
        self.decode_node(node.l_son, keys)
        self.decode_node(node.r_son, keys)

    def decode_tree(self, keys):
        self.decode_node(self.tree.root, keys)
