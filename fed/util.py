from tqdm.auto import tqdm
import argparse
import os
import time
from pathlib import Path


class MyParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--dataset', default='covtype')
        self.parser.add_argument('--nmin', default=500)
        self.parser.add_argument('--n_client', default=10)
        self.parser.add_argument('--n_round', default=12)
        self.parser.add_argument('--non_iid', default=True)
        self.parser.add_argument('--n_class', default=2)
        self.parser.add_argument('--dp', default=True)
        self.parser.add_argument('--gamma', default=0.04)
        self.parser.add_argument('--zeta', default=1.0)
        self.parser.add_argument('--record', default=False)
        self.parser.add_argument('--data_path', default=r'D:\work\py\Decision-Tree-CART\data')
        self.args = self.parser.parse_args()

    def __getitem__(self, item):
        return self.args.__getattribute__(item)

    def __getattr__(self, item):
        return self.args.__getattribute__(item)


ARGS = MyParser()
BASE_PATH = Path(os.path.realpath(__file__)).parent.parent
DATA_PATH = Path(ARGS.data_path)
_time_str = time.strftime("%m-%d-%H.%M", time.localtime())
LOG_FP = open(BASE_PATH / f'result-{_time_str}.txt', 'w', encoding='utf-8')


def log(*x):
    print(*x)
    if len(x) == 1 and isinstance(x[0], str):
        LOG_FP.write(x[0] + '\n')


log(f'Dataset: {ARGS.dataset}')
log(f'Params: lambda: {ARGS.gamma}\tzeta: {ARGS.zeta}')
log(f'#participant_type: {ARGS.n_client}\t#Epoch: {ARGS.n_round}')
