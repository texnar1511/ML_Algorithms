import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from typing import Callable
import random

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
            samples: int = 0,
            targets: pd.Series = None):
        self.column: str = column
        self.split: float = split
        self.ig: float = ig
        self.left: Tree = left
        self.right: Tree = right
        self.kind: str = kind
        self.proba: float = proba
        self.samples: int = samples
        self.targets: pd.Series = targets
    
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

    def bypass(self):
        node = self
        if node.kind == 'leaf':
            print(node.proba)
        else:
            node.left.bypass()
            node.right.bypass()
            
    def change_probas(self, y_true: pd.Series, y_pred: pd.Series, loss: str, leafs_reg: float) -> None:
        node = self
        if node.kind == 'leaf':
            ans = (node.targets + y_true - y_pred).dropna() - node.targets
            node.proba = ans.mean() if loss == 'MSE' else ans.median()
            node.proba += leafs_reg
        else:
            node.left.change_probas(y_true, y_pred, loss, leafs_reg)
            node.right.change_probas(y_true, y_pred, loss, leafs_reg)
        
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
            node = Tree(kind = 'leaf', proba = y.mean(), samples = len(y), targets = y)
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
    
    def __init__(self, n_estimators: int = 10, 
                 learning_rate: float = 0.1, 
                 max_depth: int = 5, 
                 min_samples_split: int = 2, 
                 max_leafs: int = 20, 
                 bins: int = 16, 
                 loss: str = 'MSE', 
                 metric: str = None, 
                 max_features: float = 0.5, 
                 max_samples: float = 0.5, 
                 reg: float = 0.1,
                 random_state = 42
                 ) -> None:
        self.n_estimators: int = n_estimators
        self.learning_rate: float = learning_rate
        self.max_depth: int = max_depth
        self.min_samples_split: int = min_samples_split
        self.max_leafs: int = max_leafs
        self.bins: int = bins
        self.loss: str = loss
        self.pred_0: float = 0.0
        self.trees: list[MyTreeReg] = []
        self._metrics: dict[str, Callable[[pd.Series, pd.Series], float]] = {'MSE': self.MSEMetric, 'MAE': self.MAEMetric, 'RMSE': self.RMSEMetric, 'R2': self.R2Metric, 'MAPE': self.MAPEMetric}
        self.metric = self._metrics.get(metric, metric)
        self.best_score = float('inf')
        self.max_features = max_features
        self.max_samples = max_samples
        self.reg = reg
        self.fi = {}
        self.random_state = random_state
        
    def __str__(self) -> str:
        ans = 'MyBoostReg class:'
        for key in self.__dict__:
            ans += f' {key}={self.__dict__[key]},'
        return ans[:-1] 
    
    def MSELoss(self, y_pred: pd.Series, y_true: pd.Series) -> pd.Series:
        return ((y_pred - y_true) ** 2).dropna()
    
    def MSEGradient(self, y_pred: pd.Series, y_true: pd.Series) -> pd.Series:
        return ((y_pred - y_true) * 2).dropna()
    
    def MAELoss(self, y_pred: pd.Series, y_true: pd.Series) -> pd.Series:
        return ((y_pred - y_true).abs()).dropna()
    
    def MAEGradient(self, y_pred: pd.Series, y_true: pd.Series) -> pd.Series:
        return pd.Series(np.sign(y_pred - y_true)).dropna()
    
    def MSEMetric(self, y_pred: pd.Series, y_true: pd.Series) -> float:
        return ((y_pred - y_true) ** 2).mean()
    
    def MAEMetric(self, y_pred: pd.Series, y_true: pd.Series) -> float:
        return (y_pred - y_true).abs().mean()
    
    def RMSEMetric(self, y_pred: pd.Series, y_true: pd.Series) -> float:
        return ((y_pred - y_true) ** 2).mean() ** 0.5
    
    def R2Metric(self, y_pred: pd.Series, y_true: pd.Series) -> float:
        return 1 - ((y_pred - y_true) ** 2).sum() / ((y_true.mean() - y_true) ** 2).sum()
    
    def MAPEMetric(self, y_pred: pd.Series, y_true: pd.Series) -> float:
        return (1 - y_pred / y_true).abs().mean() * 100
    
    def Stopping(self, scores, count):
        return len([i for i in scores[-count:] if i >= scores[-count - 1]]) == count if len(scores) >= count + 1 else False 
            
            
    
    def fit(self, X: pd.DataFrame, y: pd.Series, X_eval: pd.DataFrame = None, y_eval: pd.Series = None, early_stopping: int = None, verbose: int = None) -> None:
        for column in X:
            self.fi[column] = 0.0
        random.seed(self.random_state)
        self.pred_0 = y.mean() if self.loss == 'MSE' else y.median()    
        F = self.pred_0
        scores = []
        for i in range(1, self.n_estimators + 1):
            cols_idx = random.sample(list(X.columns), round(X.shape[1] * self.max_features))
            rows_idx = random.sample(range(X.shape[0]),  round(X.shape[0] * self.max_samples))
            X_s = X[cols_idx].iloc[rows_idx]
            y_s = y.iloc[rows_idx]
            gradient = self.MSEGradient(F, y_s) if self.loss == 'MSE' else self.MAEGradient(F, y_s)
            tree_reg = MyTreeReg(self.max_depth, self.min_samples_split, self.max_leafs, self.bins, len(X))
            tree_reg.fit(X_s, -gradient)
            tree_reg.tree.change_probas(y_s, F, self.loss, self.reg * sum([tree.leafs_cnt for tree in self.trees]))
            for k, v in tree_reg.fi.items():
                self.fi[k] += v
            self.trees += [tree_reg]
            loss_value = self.MSELoss(F, y_s) if self.loss == 'MSE' else self.MAELoss(F, y_s)
            self.best_score = loss_value.mean() if not self.metric else self.metric(tree_reg.predict(X), y)
            if early_stopping is not None and X_eval is not None and y_eval is not None:
                eval_loss = self.MSELoss(F, y_eval) if self.loss == 'MSE' else self.MAELoss(F, y_eval)
                eval_score = eval_loss.mean() if not self.metric else self.metric(tree_reg.predict(X_eval), y_eval)
                scores += [eval_score]
            if verbose and not i % verbose:
                print(f'{i}. Loss[{self.loss}]: {loss_value.mean()}' + (f' | {self.metric.__name__[:-6]}: {self.best_score}' if self.metric else '') + (f' | eval: {eval_score}' if self.metric and early_stopping is not None and X_eval is not None and y_eval is not None else ''))
            F += tree_reg.predict(X) * (self.learning_rate if not callable(self.learning_rate) else self.learning_rate(i))
            print(scores)
            print(self.Stopping(scores, early_stopping))
            if self.Stopping(scores, early_stopping):
                self.trees = self.trees[:-early_stopping]
                self.best_score = scores[-early_stopping - 1]
                break
        else:
            loss_value = self.MSELoss(F, y) if self.loss == 'MSE' else self.MAELoss(F, y)
            self.best_score = loss_value.mean() if not self.metric else self.metric(self.predict(X), y)
            
    def predict(self, X: pd.DataFrame):
        ans = 0.0
        for i, tree_reg in enumerate(self.trees):
            ans += tree_reg.predict(X) * (self.learning_rate if not callable(self.learning_rate) else self.learning_rate(i + 1))
        return ans + self.pred_0
        
X, y = make_regression(n_samples = 100, n_features = 4, n_informative = 10, noise = 15, random_state = 42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

X_eval, y_eval = make_regression(n_samples = 50, n_features = 4, n_informative = 10, noise = 15, random_state = 42)
X_eval = pd.DataFrame(X_eval)
y_eval = pd.Series(y_eval)
X_eval.columns = [f'col_{col}' for col in X_eval.columns]
    
a = MyBoostReg(n_estimators = 20, loss = 'MAE', metric = 'MAPE')
a.fit(X, y, X_eval, y_eval, early_stopping = 4, verbose = 3)
print(a.pred_0)
print(a.predict(X).sum())
print(a.best_score)
print(a.fi)

        
    