import numpy as np
from collections import Counter

class DT_node():
    def __init__(self, ans=None, feature=None, son=None):
        if feature: self.feature = (int(feature[0]), feature[1])
        self.ans      =   ans
        self.son      =   son

class DecisionTree():
    def __init__(self, data=None, min_gain=0.8):
        self.min_gain = min_gain
        if data is not None: self.fit(data)

    def fit(self, data):
        self.root = self.build(data)
        # self.prune(self.root)

    def gini(self, y):
        counter = Counter(y)
        res, n = 1.0, len(y)
        for num in counter.values():
            p = num / n
            res -= p**2
        return res

    def split(self, data, i, v):
        f0 = [x[i] <= v for x in data[0]]
        f1 = [not x for x in f0]
        return (data[0][f0], data[1][f0]), (data[0][f1], data[1][f1])

    def build(self, data):
        cur_gini  = self.gini(data[1])
        n = len(data)

        best_gain    = 0.
        best_feature = None
        best_split   = None

        for i in range(len(data[0][0])):
            vals = set(data[0][:, i])
            for v in vals:
                s1, s2 = self.split(data, i, v)
                p = float(len(s1[1])) / n
                gain = cur_gini - p*self.gini(s1[1]) - (1-p)*self.gini(s2[1])
                if gain > best_gain and len(s1[1]) and len(s2[1]):
                    best_gain    = gain
                    best_feature = (i, v)
                    best_split   = (s1, s2)

        if best_gain > 0:
            return DT_node(feature=best_feature, 
                           son = [self.build(best_split[0]), self.build(best_split[1])])
        else:
            return DT_node(ans=Counter(data[1]))

    def pred(self, node, x):
        if node.ans is not None:
            res_counter = Counter(node.ans)
            return max(res_counter, key=res_counter.get)
        return self.pred(node.son[int(x[node.feature[0]] > node.feature[1])], x) 

    def predict(self, X):
        return [self.pred(self.root, x) for x in X]

    def score(self, X, y):
        res = self.predict(X)
        n, m = 0, len(y)
        for i in range(m): n += int(res[i]==y[i])
        print('{}/{}, Acc: {:.2f}%'.format(int(n), m, (100. * n) / m))

    def prune(self, node):
        if not node.son: return
        self.prune(node.son[0])
        self.prune(node.son[1])

        if node.son[0].ans is not None and node.son[1].ans is not None:
            # print(type(node.son[0].ans), type(node.son[1].ans))
            b0, b1 = [], []
            for k in node.son[0].ans: b0 += [k] * node.son[0].ans[k]
            for k in node.son[1].ans: b1 += [k] * node.son[1].ans[k]

            bb    = b0 + b1
            # print(type(b0), type(b1), type(bb))
            p     = float(len(b0)) / len(bb)
            delta = self.gini(bb) - p*self.gini(b0) - (1-p)*self.gini(b1)

            if delta < self.min_gain:
                node.son = None
                node.ans = Counter(bb)
