import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
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
            
    def change_probas(self, y: pd.Series, p: pd.Series, leafs_reg: int) -> None:
        node = self
        if node.kind == 'leaf':
            y = (node.targets + y).dropna() - node.targets
            p = (node.targets + p).dropna() - node.targets
            node.proba = (y - p).sum() / (p * (1 - p)).sum() + leafs_reg
        else:
            node.left.change_probas(y, p, leafs_reg)
            node.right.change_probas(y, p, leafs_reg)
        
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

class MyBoostClf():
    
    def __init__(self,
                 n_estimators: int = 10,
                 learning_rate: float = 0.1,
                 max_depth: int = 5,
                 min_samples_split: int = 2,
                 max_leafs: int = 20,
                 bins: int = 16,
                 metric: str = None,
                 max_features: float = 0.5,
                 max_samples: float = 0.5,
                 random_state = 42,
                 reg: float = 0.1
                 ) -> None:
        
        self.n_estimators: int = n_estimators
        self.learning_rate: float = learning_rate
        self.max_depth: int = max_depth
        self.min_samples_split: int = min_samples_split
        self.max_leafs: int = max_leafs
        self.bins: int = bins
        self.pred_0 = 0.0
        self.trees: list[MyTreeReg] = []
        self.eps = 1e-15
        self.best_score = 0.0
        self.metric = metric
        self._metrics = {'accuracy': self.Accuracy, 'precision': self.Precision, 'recall': self.Recall, 'f1': self.F1, 'roc_auc': self.ROC_AUC}
        self.metric_fun = self._metrics.get(metric, metric)
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.reg = reg
        self.fi = {}
        
    def __str__(self) -> str:
        ans = 'MyBoostClf class:'
        for key in self.__dict__:
            ans += f' {key}={self.__dict__[key]},'
        return ans[:-1]
    
    def LogLoss(self, y: pd.Series, log_odds: pd.Series) -> pd.Series:
        return -(y * log_odds - np.log(1 + np.exp(log_odds) + self.eps)).mean()
    
    def Accuracy(self, y: pd.Series, y_pred: pd.Series):
        return (y == y_pred).mean()
    
    def Precision(self, y: pd.Series, y_pred: pd.Series):
        return (y * y_pred).sum() / y_pred.sum()
    
    def Recall(self, y: pd.Series, y_pred: pd.Series):
        return (y * y_pred).sum() / y.sum()
    
    def F1(self, y: pd.Series, y_pred: pd.Series):
        return 2 * (y * y_pred).sum() / (y.sum() + y_pred.sum())
    
    def ROC_AUC(self, y: pd.Series, y_pred_proba: pd.Series):
        y_pred_proba = y_pred_proba.round(10)
        df = pd.concat([y_pred_proba, y], axis = 1)
        df.columns = [0, 1]
        df = df.sort_values([0, 1], ascending = [False, False]).reset_index()
        roc_auc = 0.0
        for i in df.index:
            if df.iloc[i, 2] == 0:
                roc_auc += df.iloc[:i, 2][df.iloc[:i, 1] > df.iloc[i, 1]].sum()
                roc_auc += df.iloc[:i, 2][df.iloc[:i, 1] == df.iloc[i, 1]].sum() / 2
        s = y.sum()
        return roc_auc / s / (len(y) - s)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = None) -> None:
        for column in X:
            self.fi[column] = 0.0
        random.seed(self.random_state)
        X.reset_index(drop = True, inplace = True)
        y.reset_index(drop = True, inplace = True)
        p = y.mean()
        log_odds = np.log(p / (1 - p))
        self.pred_0 = log_odds
        p = pd.Series(np.full(len(y), p))
        log_odds = pd.Series(np.full(len(y), log_odds))
        for i in range(1, self.n_estimators + 1):
            p = pd.Series(np.exp(log_odds) / (1 + np.exp(log_odds)))
            r = y - p
            cols_idx = random.sample(list(X.columns), round(self.max_features * X.shape[1]))
            rows_idx = random.sample(range(X.shape[0]), round(self.max_samples * X.shape[0]))
            X_s = X[cols_idx].iloc[rows_idx]
            r_s = r.iloc[rows_idx]
            treeReg = MyTreeReg(self.max_depth, self.min_samples_split, self.max_leafs, self.bins, len(X))
            treeReg.fit(X_s, r_s)
            treeReg.tree.change_probas(y, p, self.reg * sum([trg.leafs_cnt for trg in self.trees]))
            for k, v in treeReg.fi.items():
                self.fi[k] += v
            self.trees += [treeReg]
            log_odds += (self.learning_rate if not callable(self.learning_rate) else self.learning_rate(i)) * treeReg.predict(X)
            loss = self.LogLoss(y, log_odds)
            if self.metric is not None:
                metric_value = self.metric_fun(y, self.predict_proba(X)) if self.metric == 'roc_auc' else self.metric_fun(y, self.predict(X))
            if verbose is not None and not i % verbose:
                print(f'{i}: {loss}' + f' | {self.metric}: {metric_value}' if self.metric is not None else '')
        self.best_score = (self.metric_fun(y, self.predict_proba(X)) if self.metric == 'roc_auc' else self.metric_fun(y, self.predict(X))) if self.metric is not None else loss
                
    def predict_proba(self, X: pd.DataFrame):
        X.reset_index(drop = True, inplace = True)
        ans = 0.0
        for i, treeReg in enumerate(self.trees):
            ans += treeReg.predict(X) * (self.learning_rate if not callable(self.learning_rate) else self.learning_rate(i + 1))
        ans += self.pred_0
        return pd.Series(np.exp(ans) / (1 + np.exp(ans)))
    
    def predict(self, X: pd.DataFrame):
        X.reset_index(drop = True, inplace = True)
        ans = 0.0
        for i, treeReg in enumerate(self.trees):
            ans += treeReg.predict(X) * (self.learning_rate if not callable(self.learning_rate) else self.learning_rate(i + 1))
        ans += self.pred_0
        return (pd.Series(np.exp(ans) / (1 + np.exp(ans))) > 0.5).astype(int)
        
        
X, y = make_classification(n_samples=100, n_features=5, n_informative=3, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

a = MyBoostClf(metric = 'accuracy')
a.fit(X, y, 1)
#print(a.predict_proba(X))
#print(a.predict(X))