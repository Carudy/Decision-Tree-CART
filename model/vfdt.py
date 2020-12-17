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
        self.nijk = dd(lambda: dd(lambda: dd(int)))

    def add_children(self, split_feature, split_value, left, right):
        self.split_feature =  split_feature
        self.split_value   =  split_value
        self.data_type = 1 if isinstance(split_value, list) else 0  # 0 : numbers  1 : discrete
        self.left_child    =  left
        self.right_child   =  right
        left.parent = right.parent = self
        del self.nijk

    def is_leaf(self):
        return self.left_child is None and self.right_child is None

    def judge_left(self, value):
        if self.data_type == 0:  return value <= self.split_value
        return value in self.split_value[0]

    # recursively trace down the tree to distribute data examples to corresponding leaves
    def sort_data(self, x):
        if self.is_leaf():  return self
        value = x[self.split_feature]
        return self.left_child.sort_data(x) if self.judge_left(value) else self.right_child.sort_data(x)

    # the most frequent class
    def most_frequent(self):
        if bool(self.label_freq): return max(self.label_freq, key=self.label_freq.get) 
        return self.parent.most_frequent()

    # update leaf stats in order to calculate gini
    def update_stats(self, x, y):
        for key in range(self.num_feature): self.nijk[key][x[key]][y] += 1
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
        min_0 = min_1 = 1
        split_value = None
        Xa = ''
        
        for feature in range(self.num_feature):
            njk = self.nijk[feature]
            gini, value = self.gini(njk)
            if gini < min_0:
                min_1        =  min_0
                min_0        =  gini
                Xa           =  feature
                split_value  =  value
            elif gini < min_1:
                min_1 = gini

        epsilon = self.hoeffding_bound(delta)
        g_X0 = self.calc_gini(self.label_freq)
        if (min_0 < g_X0) and ((min_1 - min_0 > epsilon) or (tau != 0 and min_1 - min_0 < epsilon < tau)):
            return [Xa, split_value]
        return None

    def hoeffding_bound(self, delta):
        n = self.tot_data << 1
        R = np.log(len(self.label_freq))
        return np.sqrt(R * R * np.log(1/delta) / n)

    # gini(D, F=f) = |D1|/|D|*gini(D1) + |D2|/|D|*gini(D2)
    def gini(self, njk):
        label_freq = self.label_freq
        D  = self.tot_data
        m1 = 1    # minimum gini
        Xa_value = None
        feature_values = list(njk.keys()) # list() is essential

        if not isinstance(feature_values[0], str):  # numeric feature values
            sort = np.array(sorted(feature_values))
            split = (sort[:-1] + sort[1:]) * 0.5
            D1_cf = {j: 0 for j in label_freq.keys()}
            for index in range(len(split)):
                nk = njk[sort[index]]
                for j in nk: D1_cf[j] += nk[j]
                D2_cf = {k : v - D1_cf[k] if k in D1_cf else v for k, v in label_freq.items()}
                D1 = sum(D1_cf.values())
                D2 = D - D1
                g_d1 = self.calc_gini(D1_cf)
                g_d2 = self.calc_gini(D2_cf)
                g = g_d1*D1/D + g_d2*D2/D
                if g < m1:
                    m1 = g
                    Xa_value = split[index]
            return [m1, Xa_value]

# very fast decision tree class, i.e. hoeffding tree
class Vfdt:
    def __init__(self, num_feature, delta=0.01, nmin=100, tau=0.1):
        """
          num_feature : number of features
          delta       : used to compute hoeffding bound, error rate
          nmin        : to limit the G computations, re-calc every nmin data samples
          tau         : to deal with ties
        """
        self.delta = delta
        self.nmin  = nmin
        self.tau   = tau
        self.root  = VfdtNode(num_feature)
        self.num_feature = num_feature

    # update the tree by adding one or many training example(s)
    def update(self, X, y):
        if isinstance(y, int): 
            self.update_single(X, y)
        else:
            for x, _y in zip(X, y): self.update_single(x, _y)

    # update the tree by adding one training example
    def update_single(self, x, _y):
        node = self.root.sort_data(x)
        node.update_stats(x, _y)
        result = node.attempt_split(self.delta, self.nmin, self.tau)
        if result is not None:
            feature = result[0]
            value   = result[1]
            self.node_split(node, feature, value)

    # split node, produce children
    def node_split(self, node, split_feature, split_value):
        left  = VfdtNode(self.num_feature)
        right = VfdtNode(self.num_feature)
        node.add_children(split_feature, split_value, left, right)

    # predict test example's classification
    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def predict_single(self, x):
        leaf = self.root.sort_data(x)
        return leaf.most_frequent()