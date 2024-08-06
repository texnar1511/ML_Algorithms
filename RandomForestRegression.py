import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
import random
import time
import multiprocessing as mp
from typing import Tuple

class Tree():
    
    def __init__(
            self, 
            column: str = 'col', 
            split: float = 0.0, 
            ig: float = 0.0,
            left = None, 
            right = None, 
            kind: str = 'node', 
            proba: float = 0.0, 
            samples: int = 0):
        self.column = column
        self.split = split
        self.ig = ig
        self.left: Tree = left
        self.right: Tree = right
        self.kind = kind
        self.proba = proba
        self.samples = samples
    
    def __str__(self, prefix = '', count = 0, addition = True, end = True, width = 7):
        root = f'{self.column} > {self.split} | samples = {self.samples}' if self.kind == 'node' else f'leaf_{"left" if addition else "right"} = {self.proba} | samples = {self.samples}' if self.kind == 'leaf' else ''
        new_prefix = '' if count == 0 else prefix + '│' + ' ' * width if addition and end else prefix + ' ' + ' ' * width
        left = '' if self.left is None else '\n' + self.__class__.__str__(self.left, new_prefix, count + 1, True, self.right is not None)
        right = '' if self.right is None else '\n' + self.__class__.__str__(self.right, new_prefix, count + 1, False, False)
        symbol = '├' if end else '└' 
        suffix = '' if count == 0 else symbol + '─' * width
        return prefix + suffix + root + left + right
    
    def depth(self):
        left_depth = self.left.depth() if self.left else 0
        right_depth = self.right.depth() if self.right else 0
        return 1 + max(left_depth, right_depth)
    
    def proba_sum(self):
        return self.proba + (0.0 if self.left is None else self.left.proba_sum()) + (0.0 if self.right is None else self.right.proba_sum())
    
    def traversal(self, row: pd.Series):
        node = self
        while node.kind == 'node':
            node = node.left if row[node.column] <= node.split else node.right
        return node.proba

class MyTreeReg():
    '''
        Decision tree regression
    '''

    def __init__(self, max_depth = 5, min_samples_split = 2, max_leafs = 20, bins = None, total_samples = 1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max(2, max_leafs)
        self.eps = 1e-15
        self.leafs_cnt = 0
        self.tree = Tree()
        self.bins = bins
        self.fi = {}
        self.total_samples = total_samples

    def __str__(self):
        ans = 'MyTreeReg class:\n'
        for key in self.__dict__:
            prefix = '{\n' if isinstance(self.__dict__[key], Tree) else ''
            suffix = '\n},\n' if isinstance(self.__dict__[key], Tree) else ',\n'
            ans += f'{key}={prefix}{self.__dict__[key]}{suffix}'
        return ans[:-2]
    
    def print_tree(self):
        print(self.tree)
    
    def MSE(self, classes: pd.Series):
        if len(classes) == 0:
            return 0.0
        return ((classes - classes.mean()) ** 2).mean()
    
    def MSEGain(self, X: pd.DataFrame, y: pd.Series, column: str, split: float):
        left = y[X[column] <= split]
        right = y[X[column] > split]
        return self.MSE(y) - (len(left) * self.MSE(left) + len(right) * self.MSE(right)) / len(y)

    def FeatureImportance(self, X: pd.DataFrame, y: pd.Series, column: str, split: float):
        return self.MSEGain(X, y, column, split) * len(y) / self.total_samples

    def get_best_split(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        ig = float('-inf')
        col_name = ''
        split_value = 0.0
        for column in X:
            if self.bins:
                splits = self.hist[column]
            else:
                #start = time.time()
                #splits = np.convolve(np.sort(X[column].unique()), [0.5, 0.5], 'valid')
                #end = time.time()
                #print(end - start)
                ssu = np.sort(np.unique(X[column]))
                #print(X[column])
                #print(native)
                native = (ssu[1:] + ssu[:-1]) / 2
            for split in splits:
                tmp_ig = self.MSEGain(X, y, column, split) 
                if tmp_ig > ig:
                    ig = tmp_ig
                    col_name = column
                    split_value = split
        return col_name, split_value, ig
    
    def splits_count(self, X: pd.DataFrame) -> int:
        ans = 0
        for column in X:
            #print(self.hist[column])
            #print((self.hist[column] >= X[column].min()) & (self.hist[column] <= X[column].max()))
            ans += len(self.hist[column][(self.hist[column] > X[column].min()) & (self.hist[column] < X[column].max())])
        return ans
    
    def growth_tree(self, X: pd.DataFrame, y: pd.Series, depth: int, future_leafs: int) -> Tree:
        if (len(y.unique()) == 1 or 
            depth >= self.max_depth or 
            len(y) < self.min_samples_split or 
            self.leafs_cnt + future_leafs - self.max_leafs + 1 >= 0 or 
            (self.bins and self.splits_count(X) == 0)
            ):
            node = Tree(kind = 'leaf', proba = y.mean(), samples = len(y))
            self.leafs_cnt += 1
        else:
            node = Tree(*self.get_best_split(X, y), samples = len(y))
            self.fi[node.column] += self.FeatureImportance(X, y, node.column, node.split)
            left = X[node.column] <= node.split
            right = X[node.column] > node.split
            #X_left = X[left]
            #y_left = y[left]
            #X_right = X[right]
            #y_right = y[right]
            node.left = self.growth_tree(X[left], y[left], depth + 1, future_leafs + 1)
            node.right = self.growth_tree(X[right], y[right], depth + 1, future_leafs)
        return node
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        #self.total_samples = len(X)
        for column in X:
            self.fi[column] = 0
        if self.bins:
            h = {}
            for column in X:
                #native = np.convolve(np.sort(np.unique(X[column])), [0.5, 0.5], 'valid')
                #print(type(X[column]))
                su = np.sort(np.unique(X[column]))
                #print(X[column])
                #print(native)
                native = (su[1:] + su[:-1]) / 2
                if native.shape[0] <= self.bins - 1:
                    h[column] = native
                else:
                    _, bin_edges = np.histogram(X[column], self.bins)
                    h[column] = bin_edges[1:-1]
            self.hist = h
        self.tree = self.growth_tree(X, y, 0, 0)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        ans = []
        for index, row in X.iterrows():
            ans += [self.tree.traversal(row)]
        return pd.Series(ans)

class MyForestReg():

    def __init__(self, 
                 n_estimators = 10, 
                 max_features = 0.5, 
                 max_samples = 0.5, 
                 max_depth = 5,
                 min_samples_split = 2,
                 max_leafs = 20,
                 bins = 16,
                 random_state = 42,
                 oob_score = None,
                 parallel_train = False
                 ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.random_state = random_state
        self.leafs_cnt = 0
        self.forest: list[MyTreeReg] = []
        self.fi = {}
        self.oob_score = oob_score
        self.oob_score_ = 0
        self.metrics = {'mae': self.MAE, 'mse': self.MSE, 'rmse': self.RMSE, 'mape': self.MAPE, 'r2': self.R2}
        self.parallel_train = parallel_train

    def __str__(self):
        ans = 'MyForestReg class:'
        for key in self.__dict__:
            ans += f' {key}={self.__dict__[key]},'
        return ans[:-1]
    
    def MAE(self, y: pd.Series, y_pred: pd.Series):
        return (y - y_pred).abs().mean()
    
    def MSE(self, y: pd.Series, y_pred: pd.Series):
        return ((y - y_pred) ** 2).mean()
    
    def RMSE(self, y: pd.Series, y_pred: pd.Series):
        return (((y - y_pred) ** 2).mean()) ** (1 / 2)
    
    def MAPE(self, y: pd.Series, y_pred: pd.Series):
        return ((y - y_pred) / y).abs().mean() * 100
    
    def R2(self, y: pd.Series, y_pred: pd.Series):
        return 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    
    def loop_iter(self, Xy: Tuple[pd.DataFrame, pd.Series]) -> pd.Series:
        X, y = Xy
        cols_idx = random.sample(list(X.columns), round(X.shape[1] * self.max_features))
        rows_idx = random.sample(range(X.shape[0]), round(X.shape[0] * self.max_samples))
        X_s = X.loc[rows_idx, cols_idx]
        y_s = y.loc[rows_idx]
        tree = MyTreeReg(self.max_depth, self.min_samples_split, self.max_leafs, self.bins, len(X))
        tree.fit(X_s, y_s)
        self.forest += [tree]
        self.leafs_cnt += tree.leafs_cnt
        rows_idx_oob = list(set(range(X.shape[0])) - set(rows_idx))
        X_s_oob = X.loc[rows_idx_oob, cols_idx]
        pred_oob = tree.predict(X_s_oob)
        pred_oob.index = X_s_oob.index
        #preds_oob += [pred_oob]
        return pred_oob

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        random.seed(self.random_state)
        for column in X:
            self.fi[column] = 0
        preds_oob = []
        if self.parallel_train:
            pool = mp.Pool(mp.cpu_count())
            pred_oob = pool.map(self.loop_iter, [(X, y)] * self.n_estimators)
        else:
            for i in range(self.n_estimators):
                cols_idx = random.sample(list(X.columns), round(X.shape[1] * self.max_features))
                rows_idx = random.sample(range(X.shape[0]), round(X.shape[0] * self.max_samples))
                X_s = X.loc[rows_idx, cols_idx]
                y_s = y.loc[rows_idx]
                tree = MyTreeReg(self.max_depth, self.min_samples_split, self.max_leafs, self.bins, len(X))
                tree.fit(X_s, y_s)
                self.forest += [tree]
                self.leafs_cnt += tree.leafs_cnt
                rows_idx_oob = list(set(range(X.shape[0])) - set(rows_idx))
                X_s_oob = X.loc[rows_idx_oob, cols_idx]
                pred_oob = tree.predict(X_s_oob)
                pred_oob.index = X_s_oob.index
                preds_oob += [pred_oob]
        for column in X:
            self.fi[column] = sum([tree.fi.get(column, 0) for tree in self.forest])
        preds_oob = pd.DataFrame(preds_oob).mean(axis = 0)
        if self.oob_score:
            self.oob_score_ = self.metrics[self.oob_score](y.iloc[preds_oob.index], preds_oob)

    def predict(self, X: pd.DataFrame):
        return pd.DataFrame([tree.predict(X) for tree in self.forest]).mean(axis = 0)

X, y = make_regression(n_samples = 150, n_features = 14, n_informative = 10, noise = 15, random_state = 42)
X = pd.DataFrame(X).round(2)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]
#test = X.sample(20, random_state = 42)
    
d = {"parallel_train": True, "n_estimators": 10, "max_depth": 5, "max_samples": 0.9, "max_leafs": 15, "random_state": 42, "oob_score": "mae"}
a = MyForestReg(**d)
#print(a)
#print(X)

start = time.time()

a.fit(X, y)

end = time.time()

print(end - start)

#print(a.leafs_cnt)
#print(a.predict(X))
#print([tree.fi for tree in a.forest])
#print(a.fi)
#print(y)
#print(a.oob_preds)
print(a.oob_score_) 
