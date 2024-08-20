import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

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
                ssu = np.sort(np.unique(X[column]))
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
            node.left = self.growth_tree(X[left], y[left], depth + 1, future_leafs + 1)
            node.right = self.growth_tree(X[right], y[right], depth + 1, future_leafs)
        return node
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        for column in X:
            self.fi[column] = 0
        if self.bins:
            h = {}
            for column in X:
                su = np.sort(np.unique(X[column]))
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

class MyBoostReg():
    
    def __init__(self, n_estimators: int = 10, learning_rate: float = 0.1, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20, bins: int = 16, loss: str = 'MSE') -> None:
        self.n_estimators: int = n_estimators
        self.learning_rate: float = learning_rate
        self.max_depth: int = max_depth
        self.min_samples_split: int = min_samples_split
        self.max_leafs: int = max_leafs
        self.bins: int = bins
        self.loss: str = loss
        self.pred_0: float = 0.0
        self.trees: list[MyTreeReg] = []
        
    def __str__(self) -> str:
        ans = 'MyBoostReg class:'
        for key in self.__dict__:
            ans += f' {key}={self.__dict__[key]},'
        return ans[:-1] 
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.pred_0 = y.mean()
        F = self.pred_0
        for i in range(self.n_estimators):
            r = y - F
            tree_reg = MyTreeReg(self.max_depth, self.min_samples_split, self.max_leafs, self.bins)
            tree_reg.fit(X, r)
            self.trees += [tree_reg]
            F += self.learning_rate * tree_reg.predict(X)
            
    def predict(self, X: pd.DataFrame):
        ans = 0.0
        for tree_reg in self.trees:
            ans += tree_reg.predict(X)
        return ans * self.learning_rate + self.pred_0
        
X, y = make_regression(n_samples = 100, n_features = 4, n_informative = 10, noise = 15, random_state = 42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]
    
a = MyBoostReg()
a.fit(X, y)
print(a.pred_0)
print(a.trees)
print(a.predict(X).sum())

        
    