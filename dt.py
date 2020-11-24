import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

TYPES = ['virginica', 'versicolor', 'setosa']
TYPES_ID = {tp : i for (i, tp) in enumerate(TYPES)}

class DT_node():
    def __init__(self, ans=None, feature=None, son=None):
        if feature: self.feature = (int(feature[0]), feature[1])
        self.ans      =   ans
        self.son      =   son

class DecisionTree():
    def __init__(self, data=None):
        if data is not None: self.fit(data)

    def fit(self, data):
        self.root = self.build(data)

    def gini(self, y):
        counter = Counter(y)
        res, n = 1.0, len(y)
        for num in counter.values():
            p = num / n
            res -= p**2
        return res

    def split(self, data, i, v):
        f0 = [x[i] >= v for x in data[0]]
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
            res_counter = Counter(data[1])
            res = max(res_counter, key=res_counter.get)
            return DT_node(ans=res)

    def pred(self, node, x):
        return node.ans if node.ans is not None else self.pred(node.son[int(x[node.feature[0]] < node.feature[1])], x) 

    def predict(self, X):
        return [self.pred(self.root, x) for x in X]

    def score(self, X, y):
        res = self.predict(X)
        n, m = 0, len(y)
        for i in range(m): n += int(res[i]==y[i])
        print('{}/{}, Acc: {:.2f}%'.format(int(n), m, (100. * n) / m))


def read_data():
    data = [line.split(' ') for line in open('iris.txt').readlines()[1:]]
    X = np.array([line[1:-1] for line in data]).astype(np.float64)
    y = np.array([TYPES_ID[line[-1][:-1].strip('"')] for line in data]).astype(np.int)
    return X, y

if __name__ == '__main__':
    X, y = read_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=996)
    
    dt = DecisionTree((X_train, y_train))

    print('Training set:', end='\t')
    dt.score(X_train, y_train)  
    print('Test set:', end='\t')
    dt.score(X_test, y_test)