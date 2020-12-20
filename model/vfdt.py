'''
    Author: Dy
    Modified version of: https://github.com/doubleplusplus/incremental_decision_tree-CART-Random_Forest
    Only support continuous values at present
    Privacy-preserving Federated Incremental Decision Tree on Non-IID Data
'''
import numpy as np
from itertools import combinations
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import defaultdict as dd

# VFDT node class
class VfdtNode:
    def __init__(self, num_feature):
        self.parent         =   None
        self.left_child     =   None
        self.right_child    =   None
        self.split_feature  =   None
        self.split_value    =   None
        self.new_data       =   0
        self.tot_data       =   0
        self.label_freq     =   dd(int)
        self.num_feature    =   num_feature
        self.nijk = {i : {} for i in range(num_feature)}

    def add_children(self, split_feature, split_value, left, right):
        self.split_feature =  split_feature
        self.split_value   =  split_value
        # self.data_type = 1 if isinstance(split_value, list) else 0  # 0 : numbers  1 : discrete
        self.left_child    =  left
        self.right_child   =  right
        left.parent = right.parent = self
        del self.nijk

    def is_leaf(self):
        return self.left_child is None and self.right_child is None

    def judge_left(self, value):
        return value <= self.split_value
        # if self.data_type == 0:  return value <= self.split_value
        # return value in self.split_value[0]

    # recursively trace down the tree to distribute data examples to corresponding leaves
    def sort_data(self, x):
        if self.is_leaf():  return self
        value = x[self.split_feature]
        return self.left_child.sort_data(x) if self.judge_left(value) else self.right_child.sort_data(x)

    # the most frequent class
    def most_frequent(self):
        if bool(self.label_freq): return max(self.label_freq, key=self.label_freq.get) 
        return self.parent.most_frequent()

    def add_nijk(self, i, j, k):
        if j not in self.nijk[i] or self.nijk[i][j] is None: self.nijk[i][j] = dd(int)
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
        for j, k in arr.items(): res -= (k/n)**2
        return res

    # use Hoeffding tree model to test node split, return the split feature
    def attempt_split(self, delta, nmin, tau):
        if self.new_data < nmin:      return None
        if len(self.label_freq) == 1: return None

        self.new_data = 0
        ginis = sorted(self.gini(self.nijk[feature]) + [feature] for feature in self.nijk)
        min_0, split_value, Xa = ginis[0]
        min_1 = ginis[1][0]

        epsilon = self.hoeffding_bound(delta)
        g_X0 = self.calc_gini(self.label_freq)
        if (min_0 < g_X0) and ((min_1 - min_0 > epsilon) or (min_1 - min_0 < epsilon < tau)):
            left  = VfdtNode(self.num_feature)
            right = VfdtNode(self.num_feature)
            self.add_children(Xa, split_value, left, right)

    def hoeffding_bound(self, delta):
        n = self.tot_data << 1
        R = np.log(len(self.label_freq))
        return np.sqrt(R * R * np.log(1/delta) / n)

    # gini(D, F=f) = |D1|/|D|*gini(D1) + |D2|/|D|*gini(D2)
    def gini(self, njk):
        D, min_g, Xa_value  = self.tot_data, 1, None
        sort  = np.array(sorted(list(njk.keys())))
        split = (sort[:-1] + sort[1:]) / 2
        D1_cf = {j: 0 for j in self.label_freq.keys()}
        D2_cf = self.label_freq.copy()
        D1, D2 = 0, D
        for index in range(len(split)-1):
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
    def __init__(self, num_feature, delta=0.001, nmin=100, tau=0.1):
        self.delta = delta
        self.nmin  = nmin
        self.tau   = tau
        self.root  = VfdtNode(num_feature)
        self.num_feature = num_feature
        self.last_node = 0
        self.record_size = False

    # update the tree by adding one or many training example(s)
    def update(self, X, y):
        if isinstance(y, int): 
            self.update_single(X, y)
        else:
            for x, _y in zip(X, y): self.update_single(x, _y)

    def partial_fit(self, X, y):
        return self.update(X, y)

    # update the tree by adding one training example
    def update_single(self, x, _y):
        node = self.root.sort_data(x)
        node.update_stats(x, _y)
        node.attempt_split(self.delta, self.nmin, self.tau)
        
        if self.record_size:
            now_node = self.num_nodes()
            if now_node != self.last_node: 
                print(now_node)
                self.last_node = now_node

    # predict test example's classification
    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def predict_single(self, x):
        leaf = self.root.sort_data(x)
        return leaf.most_frequent()

    def dfs(self, node):
        if node is None: return 0
        if node.is_leaf(): return 1
        return 1 + self.dfs(node.left_child) + self.dfs(node.right_child)

    def num_nodes(self):
        return self.dfs(self.root)