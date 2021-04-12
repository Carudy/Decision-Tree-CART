'''
    Author: Dy
    Modified & fixed version of: https://github.com/doubleplusplus/incremental_decision_tree-CART-Random_Forest
    Only support continuous values at present
    Add max_depth, regional counting, etc.
'''
import numpy as np
from itertools import combinations
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import defaultdict as dd
from tqdm import tqdm

# VFDT node class
class VfdtNode:
    def __init__(self, num_feature, depth=1, max_depth=18, regional=None):
        self.parent         =   None
        self.l_son          =   None
        self.r_son          =   None
        self.split_feature  =   None
        self.split_value    =   None
        self.new_data       =   0
        self.tot_data       =   0
        self.label_freq     =   dd(int)
        self.num_feature    =   num_feature
        self.nijk = {i : {} for i in range(num_feature)}
        self.depth          =   depth
        self.max_depth      =   max_depth
        self.regional       =   regional

    def node_split(self, split_feature, split_value):
        self.split_feature =  split_feature
        self.split_value   =  split_value
        self.l_son   =  VfdtNode(self.num_feature, 
            depth=self.depth+1, 
            max_depth=self.max_depth, 
            regional=self.regional)
        self.r_son   =  VfdtNode(self.num_feature, 
            depth=self.depth+1, 
            max_depth=self.max_depth, 
            regional=self.regional)
        self.l_son.parent = self.r_son.parent = self
        del self.nijk

    def is_leaf(self):
        return self.l_son is None and self.r_son is None

    def judge_left(self, value):
        return value <= self.split_value

    # recursively trace down the tree to distribute data examples to corresponding leaves
    def sort_to_leaf(self, x):
        if self.is_leaf():  return self
        value = x[self.split_feature]
        return self.l_son.sort_to_leaf(x) if self.judge_left(value) else self.r_son.sort_to_leaf(x)

    # the most frequent class
    def most_frequent(self):
        if bool(self.label_freq): return max(self.label_freq, key=self.label_freq.get) 
        return self.parent.most_frequent()

    def add_nijk(self, i, j, k):
        if j not in self.nijk[i]: self.nijk[i][j] = {}
        if k not in self.nijk[i][j]: self.nijk[i][j][k] = 0
        self.nijk[i][j][k] += 1

    # update leaf stats in order to calculate gini
    def update_stats(self, x, y):
        for key in range(self.num_feature): self.add_nijk(key, x[key], y)
        self.tot_data += 1
        self.new_data += 1
        self.label_freq[y]  += 1

    # gini(D) = 1 - Sum(pi^2)
    def calc_gini(self, arr):
        res, n = 1, sum(arr.values())
        if n==0: 
            print('GG')
            return 1
        for j, k in arr.items(): res -= (k/n)**2
        return res

    # use Hoeffding tree model to test node split, return the split feature
    def attempt_split(self, delta, nmin, tau):
        if self.depth >= self.max_depth > 0: return
        if self.new_data < nmin:      return
        if len(self.label_freq) == 1: return

        self.new_data = 0
        if self.regional is not None: 
            for feature in range(self.num_feature): self.maintain(feature)

        ginis = sorted(self.gini(feature) + [feature] for feature in self.nijk)
        min_0, split_value, Xa = ginis[0]
        min_1 = ginis[1][0]

        epsilon = self.hoeffding_bound(delta)
        g_X0 = self.calc_gini(self.label_freq)

        if (min_0 < g_X0) and ((min_1 - min_0 > epsilon) or (min_1 - min_0 < epsilon < tau)):
        # if (min_0 < g_X0) and ((min_1 - min_0 > epsilon) or (min_0 / min_1 > 1 - tau)):
            self.node_split(Xa, split_value)

    def hoeffding_bound(self, delta):
        n = self.tot_data << 1
        R = np.log(len(self.label_freq))
        return np.sqrt(R * R * np.log(1/delta) / n)

    def num_vals(self):
        if not self.is_leaf():
            return self.l_son.num_vals() + self.r_son.num_vals()
        ret = 0

        if self.regional is not None:
            for feature in range(self.num_feature): self.maintain(feature)

        for fea in self.nijk:
            ret += len(list(self.nijk[fea].keys()))
        return ret

    # Regional counting 
    # ddos 256 sen 128 covtype 200 
    def maintain(self, feature):
        # if self.depth >= self.max_depth * 0.66: return
        vals = sorted(list(self.nijk[feature].keys()))
        if len(vals) < 256: return
        pace = self.regional
        v0 = np.array(vals)
        v0 = min(v0[1:] - v0[:-1])
        if v0 > pace: return
        # print(len(vals), end='_')
        i, n_vals = 0, len(vals)
        new_jk = {}
        # print(vals[:2], end=' ')
        # print(vals[-2:], end=' ')
        while i < n_vals:
            j, d = i, dd(int)
            for k in self.nijk[feature][vals[j]]: d[k] += self.nijk[feature][vals[j]][k]
            r = vals[i] + pace
            while j+1 < n_vals and vals[j+1] <= r: 
                j += 1
                for k in self.nijk[feature][vals[j]]: d[k] += self.nijk[feature][vals[j]][k]
            m = np.average(vals[i:j+1])
            i = j + 1
            new_jk[m] = {}
            for k in self.label_freq: 
                if d[k]>0: new_jk[m][k] = d[k]
        del self.nijk[feature]
        self.nijk[feature] = new_jk
        # vals = sorted(list(self.nijk[feature].keys()))
        # print(len(vals), end=';')

    # gini(D, F=f) = |D1|/|D|*gini(D1) + |D2|/|D|*gini(D2)
    def gini(self, feature):
        njk = self.nijk[feature]
        D, min_g, Xa_value  = self.tot_data, 1, None
        sort  = np.array(sorted(list(njk.keys())))
        split = (sort[:-1] + sort[1:]) / 2
        # print(len(split))
        D1_cf = {j: 0 for j in self.label_freq.keys()}
        D2_cf = self.label_freq.copy()
        D1, D2 = 0, D
        for index in range(len(split)):
            nk = njk[sort[index]]
            for j in nk: 
                D1_cf[j] += nk[j]
                D2_cf[j] -= nk[j]
                D1 += nk[j]
            D2 = D - D1
            g = self.calc_gini(D1_cf) * D1 / D + self.calc_gini(D2_cf) * D2 / D
            if g < min_g:
                min_g = g
                Xa_value = split[index]
        return [min_g, Xa_value]

# very fast decision tree class, i.e. hoeffding tree
class Vfdt:
    def __init__(self, num_feature, delta=1e-2, nmin=200, tau=0.05, max_depth=-1, regional=None):
        self.delta = delta
        self.nmin  = nmin
        self.tau   = tau
        self.root  = VfdtNode(num_feature, max_depth=max_depth, regional=regional)
        self.num_feature = num_feature
        self.last_node = 0
        self.progress = True
        self.T = 100000
        self.vals = []

    def update(self, X, y):
        if isinstance(y, int): 
            self.update_single(X, y)
        else:
            if self.progress:
                data = list(zip(X, y))
                for i in tqdm(range(1, len(data)+1)):
                    x, _y = data[i-1]
                    self.update_single(x, _y)
                    if i % self.T == 0: self.vals.append(self.root.num_vals())
            else:
                for x, _y in zip(X, y): self.update_single(x, _y)
        # print('num vals: {}'.format(self.root.num_vals()))
        print(self.vals)

    def partial_fit(self, X, y):
        return self.update(X, y)

    def update_single(self, x, _y):
        node = self.root.sort_to_leaf(x)
        node.update_stats(x, _y)
        node.attempt_split(self.delta, self.nmin, self.tau)

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def predict_single(self, x):
        leaf = self.root.sort_to_leaf(x)
        return leaf.most_frequent()

