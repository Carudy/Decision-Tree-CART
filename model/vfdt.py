import numpy as np
from collections import defaultdict, Iterable
from tqdm.auto import tqdm


# VFDT node class
class Vnode:
    def __init__(self, depth=1, parent=None, possible_features=None, base=None):
        self.possible_features = set(possible_features)
        self.base = base
        self.parent = parent
        self.l_son = None
        self.r_son = None
        self.split_feature = None
        self.split_value = None
        self.new_data = 0
        self.tot_data = 0
        self.label_freq = defaultdict(int)
        self.nijk = {i: {} for i in self.base.attrs}
        self.depth = depth

    def node_split(self, split_feature, split_value):
        self.split_feature = split_feature
        self.split_value = split_value
        new_possible_features = self.possible_features.copy()
        new_possible_features.remove(split_feature)
        self.l_son = Vnode(depth=self.depth + 1, parent=self, base=self.base, possible_features=new_possible_features)
        self.r_son = Vnode(depth=self.depth + 1, parent=self, base=self.base, possible_features=new_possible_features)
        del self.nijk

    def is_leaf(self):
        return self.l_son is None and self.r_son is None

    def check_branch(self, value):
        return value <= self.split_value

    def sort_to_leaf(self, x):
        if self.is_leaf():
            return self
        value = x[self.split_feature]
        return self.l_son.sort_to_leaf(x) if self.check_branch(value) else self.r_son.sort_to_leaf(x)

    def most_frequent(self):
        if bool(self.label_freq):
            return max(self.label_freq, key=self.label_freq.get)
        return self.parent.most_frequent()

    def add_nijk(self, i, j, k):
        if j not in self.nijk[i]:
            self.nijk[i][j] = {}
        if k not in self.nijk[i][j]:
            self.nijk[i][j][k] = 0
        self.nijk[i][j][k] += 1

    def update_statistics(self, x, y):
        for key in self.base.attrs:
            self.add_nijk(key, x[key], y)
        self.tot_data += 1
        self.new_data += 1
        self.label_freq[y] += 1

    # gini(D) = 1 - Sum(pi^2)
    def calc_gini(self, arr):
        res, n = 1, sum(arr.values())
        if n == 0:
            print('Wrong arr for gini.')
            return 1
        for j, k in arr.items(): res -= (k / n) ** 2
        return res

    # use Hoeffding tree model to test node split, return the split feature
    def attempt_split(self):
        if (self.depth >= self.base.max_depth > 0) or (self.new_data < self.base.nmin) or (len(self.label_freq) <= 1):
            return
        self.new_data = 0
        self.regional_count()

        ginis = sorted([self.gini(feature) + [feature] for feature in self.possible_features if feature in self.nijk])
        g_best, split_value, split_attr = ginis[0]
        g_second = ginis[1][0]

        epsilon = self.hoeffding_bound(self.base.delta)
        g_empty = self.calc_gini(self.label_freq)

        if (g_best != g_empty) and ((g_second - g_best > epsilon) or (g_second - g_best < epsilon < self.base.tau)):
            self.node_split(split_attr, split_value)

    def hoeffding_bound(self, delta):
        n = self.tot_data << 1
        R = np.log(len(self.label_freq))
        return np.sqrt(R * R * np.log(1 / delta) / n)

    def num_vals(self):
        if not self.is_leaf():
            return self.l_son.num_vals() + self.r_son.num_vals()
        ret = 0
        self.regional_count()
        for fea in self.nijk:
            ret += len(list(self.nijk[fea].keys()))
        return ret

    # Regional counting
    # ddos 256 sen 128 covtype 200
    def regional_count(self):
        if not self.base.regional: return
        features = list(self.nijk.keys())
        for feature in features:
            vals = sorted(list(self.nijk[feature].keys()))
            if len(vals) < 256: return
            pace = self.base.region_size
            v0 = np.array(vals)
            v0 = min(v0[1:] - v0[:-1])
            if v0 > pace: return
            i, n_vals = 0, len(vals)
            new_jk = {}
            while i < n_vals:
                j, d = i, defaultdict(int)
                for k in self.nijk[feature][vals[j]]: d[k] += self.nijk[feature][vals[j]][k]
                r = vals[i] + pace
                while j + 1 < n_vals and vals[j + 1] <= r:
                    j += 1
                    for k in self.nijk[feature][vals[j]]: d[k] += self.nijk[feature][vals[j]][k]
                m = np.average(vals[i:j + 1])
                i = j + 1
                new_jk[m] = {}
                for k in self.label_freq:
                    if d[k] > 0: new_jk[m][k] = d[k]
            self.nijk[feature] = new_jk

    # gini(D, F=f) = |D1|/|D|*gini(D1) + |D2|/|D|*gini(D2)
    def gini(self, feature):
        njk = self.nijk[feature]
        D, g_min, split_value = self.tot_data, 1, None
        sort = np.array(sorted(list(njk.keys())))
        keys = (sort[:-1] + sort[1:]) / 2
        D1_cf = {j: 0 for j in self.label_freq.keys()}
        D2_cf = self.label_freq.copy()
        D1 = 0
        for index in range(len(keys)):
            nk = njk[sort[index]]
            for j in nk:
                D1_cf[j] += nk[j]
                D2_cf[j] -= nk[j]
                D1 += nk[j]
            D2 = D - D1
            g = self.calc_gini(D1_cf) * D1 / D + self.calc_gini(D2_cf) * D2 / D
            if g < g_min:
                g_min = g
                split_value = keys[index]
        return [g_min, split_value]


# very fast decision tree class, i.e. hoeffding tree
class Vfdt:
    def __init__(self, attrs, delta=1e-7, nmin=500, tau=0.05, max_depth=32, regional_count=None, verbose=True):
        self.max_depth = max_depth
        if regional_count is None:
            self.regional = False
        else:
            self.regional = True
            self.region_size = regional_count
        if isinstance(attrs, int):
            self.num_feature = attrs
            self.attrs = list(range(attrs))
        else:
            self.num_feature = len(attrs)
            self.attrs = attrs
        self.verbose = verbose
        self.delta = delta
        self.nmin = nmin
        self.tau = tau
        self.root = Vnode(base=self, possible_features=attrs)
        self.last_node = 0
        self.T = 100000

    def update(self, xs, ys):
        if not isinstance(ys, Iterable):
            self.update_single(xs, ys)
        else:
            if self.verbose:
                data = list(zip(xs, ys))
                for i in tqdm(range(1, len(data) + 1)):
                    x, y = data[i - 1]
                    self.update_single(x, y)
                    # if i % self.T == 0:
                    #     self.vals.append(self.root.num_vals())
            else:
                for x, y in zip(xs, ys):
                    self.update_single(x, y)

    def partial_fit(self, xs, ys):
        return self.update(xs, ys)

    def update_single(self, x, y):
        node = self.root.sort_to_leaf(x)
        node.update_statistics(x, y)
        node.attempt_split()

    def predict(self, xs):
        return [self.predict_single(x) for x in xs]

    def predict_single(self, x):
        leaf = self.root.sort_to_leaf(x)
        return leaf.most_frequent()
