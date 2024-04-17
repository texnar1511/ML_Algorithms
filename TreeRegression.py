import pandas as pd
from sklearn.datasets import make_regression
import numpy as np

#X, y = make_regression(n_samples = 50, n_features = 20, n_informative = 2, noise = 5, random_state = 42)
#X = pd.DataFrame(X).round(2)
#y = pd.Series(y)
#X.columns = [f'col_{col}' for col in X.columns]
#test = X.sample(20, random_state = 42)

from sklearn.datasets import load_diabetes

data = load_diabetes(as_frame=True)
X, y = data['data'], data['target']

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

    def __init__(self, max_depth = 5, min_samples_split = 2, max_leafs = 20, bins = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max(2, max_leafs)
        self.eps = 1e-15
        self.leafs_cnt = 0
        self.tree = Tree()
        self.bins = bins
        self.fi = {}
        self.total_samples = 0

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
        return ((classes - classes.mean()) ** 2).sum() / len(classes)
    
    def MSEGain(self, X: pd.DataFrame, y: pd.Series, column: str, split: float):
        M0 = self.MSE(y)
        left = y[X[column] <= split]
        right = y[X[column] > split]
        M1 = self.MSE(left)
        M2 = self.MSE(right)
        return M0 - (len(left) * M1 + len(right) * M2) / len(y)

    def FeatureImportance(self, X: pd.DataFrame, y: pd.Series, column: str, split: float):
        return self.MSEGain(X, y, column, split) * len(y) / self.total_samples

    def get_best_split(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        ig = float('-inf')
        col_name = ''
        split_value = 0.0
        for column in X:
            splits = pd.Series()
            splits_native = np.convolve(np.sort(X[column].unique()), [0.5, 0.5], 'valid')
            if self.bins:
                splits_artificial = self.hist[column][(self.hist[column] > X[column].min()) & (self.hist[column] < X[column].max())]
                if len(splits_native) >= len(splits_artificial):
                    splits = splits_artificial
                else:
                    splits = splits_native
            else:
                splits = splits_native
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
        node = Tree()
        if (len(y.unique()) == 1 or 
            depth >= self.max_depth or 
            len(y) < self.min_samples_split or 
            self.leafs_cnt + future_leafs >= self.max_leafs - 1 or 
            (self.bins and self.splits_count(X) == 0)
            ):
            node = Tree(kind = 'leaf', proba = y.sum() / len(y), samples = len(y))
            self.leafs_cnt += 1
        else:
            node = Tree(*self.get_best_split(X, y), samples = len(y))
            self.fi[node.column] += self.FeatureImportance(X, y, node.column, node.split)
            left = X[node.column] <= node.split
            right = X[node.column] > node.split
            X_left = X[left]
            y_left = y[left]
            X_right = X[right]
            y_right = y[right]
            node.left = self.growth_tree(X_left, y_left, depth + 1, future_leafs + 1)
            node.right = self.growth_tree(X_right, y_right, depth + 1, future_leafs)
        return node
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.total_samples = len(X)
        for column in X:
            self.fi[column] = 0
        self.hist = pd.DataFrame()
        if self.bins:
            h = {}
            for column in X:
                _, bin_edges = np.histogram(X[column], self.bins)
                h[column] = bin_edges[1:-1]
            self.hist = pd.DataFrame(h)
        self.tree = self.growth_tree(X, y, 0, 0)

    def predict(self, X: pd.DataFrame):
        ans = []
        for index, row in X.iterrows():
            ans += [self.tree.traversal(row)]
        return pd.Series(ans)
        
a = MyTreeReg(max_depth = 15, min_samples_split = 35, max_leafs = 30, bins = None)
#print(a.get_best_split(X, y))
print(y)
a.fit(X, y)
a.print_tree()
print(a.leafs_cnt)
print(a.predict(X).sum())
print(a.fi)
#a.fit(X, y)
#a.print_tree()
#print(a.leafs_cnt)
#print(a.tree.proba_sum())
#print(a.fi)