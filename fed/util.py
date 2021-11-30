import argparse
from pathlib import Path


class MyParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--dataset', default='mushrooms')
        self.parser.add_argument('--n_client', default=10)
        self.parser.add_argument('--n_round', default=12)
        self.parser.add_argument('--data_path', default=r'D:\work\py\Decision-Tree-CART\data')
        self.args = self.parser.parse_args()

    def __getitem__(self, item):
        return self.args.__getattribute__(item)

    def __getattr__(self, item):
        return self.args.__getattribute__(item)


ARGS = MyParser()
DATA_PATH = Path(ARGS.data_path)
