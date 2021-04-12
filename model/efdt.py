'''
    Author: Dy
    Modified version of: https://github.com/doubleplusplus/incremental_decision_tree-CART-Random_Forest
    Only support continuous values at present
    Privacy-preserving Federated Incremental Decision Tree on Non-IID Data
'''

import numpy as np
from collections import defaultdict as dd

# EFDT node class
class EfdtNode:
    def __init__(self, n_features):
        self.parent       =   None
        self.l_son        =   None
        self.r_son        =   None
        self.key_feat     =   None
        self.key_value    =   None
        self.split_g      =   None
        self.new_data     =   0
        self.tot_data     =   0
        self.n_features   =   n_features
        self.nijk         =   {i: {} for i in range(n_features)}
        self.label_freq   =   dd(int)

    def add_children(self, left, right):
        self.l_son   = left
        self.r_son   = right
        left.parent  = self
        right.parent = self
        self.nijk = {i: {} for i in self.nijk.keys()}

    def is_leaf(self):
        return self.l_son is None and self.r_son is None

    def add_nijk(self, i, j, k):
        if j not in self.nijk[i]: self.nijk[i][j] = dd(int)
        self.nijk[i][j][k] += 1

    # update node stats in order to calculate Gini
    def update_stats(self, x, y):
        for key in range(self.n_features): self.add_nijk(key, x[key], y)
        self.tot_data      += 1
        self.new_data      += 1
        self.label_freq[y] += 1

    # the most frequent classification
    def most_frequent(self):
        if bool(self.label_freq): return max(self.label_freq, key=self.label_freq.get) 
        return self.parent.most_frequent()

    def calc_gini(self, arr):
        res, n = 1, sum(arr.values())
        for j, k in arr.items(): res -= (k/n)**2
        return res

    def node_split(self, split_feature, split_value):
        self.key_feat  = split_feature
        self.key_value = split_value
        self.add_children(EfdtNode(self.n_features), EfdtNode(self.n_features))

    # recursively trace down the tree
    def sort_example(self, x, y, delta, nmin, tau):
        self.update_stats(x, y)
        if self.is_leaf():
            self.attempt_split(delta, nmin, tau)
        else:
            self.re_evaluate_split(delta, nmin, tau)
            if self.is_leaf(): return
            son = self.l_son if x[self.key_feat] <= self.key_value else self.r_son
            son.sort_example(x, y, delta, nmin, tau)

    def sort_to_predict(self, x):
        if self.is_leaf(): return self
        return self.l_son if x[self.key_feat] <= self.key_value else self.r_son

    def cal_min_g(self):
        ginis = sorted(self.gini(self.nijk[feature]) + [feature] for feature in self.nijk)
        return ginis[0] + [ginis[1][0]]

    # test node split, return the split feature
    def attempt_split(self, delta, nmin, tau):
        if self.new_data < nmin or len(self.label_freq) <= 1: return
        self.new_data = 0

        g_Xa, split_value, Xa, g_Xb = self.cal_min_g()
        epsilon = self.hoeffding_bound(delta)
        g_X0 = self.calc_gini(self.label_freq)
        if (g_Xa < g_X0) and (g_Xb - g_Xa > epsilon) or (g_Xb - g_Xa < epsilon < tau):
            # self.split_g = g_Xa
            self.node_split(Xa, split_value)

    def re_evaluate_split(self, delta, nmin, tau):
        if self.new_data < nmin or len(self.label_freq) <= 1: return
        self.new_data = 0

        g_Xa, split_value, Xa, g_Xb = self.cal_min_g()
        epsilon = self.hoeffding_bound(delta)
        g_X0 = self.calc_gini(self.label_freq)
        cur_g, cur_val = self.gini(self.nijk[self.key_feat])

        if g_X0 < g_Xa:     # not split
            print('kill 0')
            self.kill_subtree()

        elif (cur_g > g_Xa) and (cur_g - g_Xa > epsilon or cur_g - g_Xa < epsilon < tau) and (Xa != self.key_feat):
            # print('kill 1')
            self.kill_subtree()
            self.split_g = g_Xa
            self.node_split(Xa, split_value)

        else:
            # print('change v')
            self.key_value = cur_val

    def kill_subtree(self):
        if not self.is_leaf():
            del self.l_son
            del self.r_son
            self.l_son = None
            self.r_son = None
            self.key_feat = None
            self.key_value = None
            self.split_g = None

    def hoeffding_bound(self, delta):
        n = self.tot_data
        R = np.log(len(self.label_freq))
        return (R * R * np.log(1/delta) / (2 * n))**0.5

    def gini(self, njk):
        D, min_g, Xa_value  = self.tot_data, 1, None
        sort  = np.array(sorted(list(njk.keys())))
        split = (sort[:-1] + sort[1:]) / 2
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


class Efdt:
    def __init__(self, features=79, delta=0.01, nmin=200, tau=0.1):
        self.features = features
        self.delta = delta
        self.nmin = nmin
        self.tau  = tau
        self.root = EfdtNode(features)
        self.n_examples_processed = 0
        self.last_node = 0
        self.record_size = False

    # update the tree by adding training example
    def update_single(self, x, y):
        self.n_examples_processed += 1
        self.root.sort_example(x, y, self.delta, self.nmin, self.tau)

        if self.record_size:
            now_node = self.num_nodes()
            if now_node != self.last_node: 
                print(now_node)
                self.last_node = now_node

    # update the tree by adding one or many training example(s)
    def update(self, X, y):
        if isinstance(y, int): 
            self.update_single(X, y)
        else:
            for x, _y in zip(X, y): self.update_single(x, _y)

    # predict test example's classification
    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def predict_single(self, x):
        leaf = self.root.sort_to_predict(x)
        return leaf.most_frequent()

    def dfs(self, node):
        if node is None: return 0
        if node.is_leaf(): return 1
        return 1 + self.dfs(node.l_son) + self.dfs(node.r_son)

    def num_nodes(self):
        return self.dfs(self.root)

    def partial_fit(self, X, y):
        return self.update(X, y)