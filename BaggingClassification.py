import numpy as np
import pandas as pd
import random
from copy import copy
from sklearn.datasets import make_classification

pd.set_option('display.max_rows', None)

class MyLogReg():
    '''
    Logistic regression
    '''

    def __init__(
            self, 
            n_iter: int = 10, 
            learning_rate: float = 0.1, 
            weights = None, 
            metric = None, 
            reg = None, 
            l1_coef: float = 0.0, 
            l2_coef: float = 0.0, 
            sgd_sample = None, 
            random_state = 42
            ):
        '''
        n_iter: int, default = 10 
            Number of Gradient Descending algorithm steps
        learning_rate: float de
        '''
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.eps = 1e-15
        self._metrics = {
            'accuracy': self.Accuracy, 
            'precision': self.Precision, 
            'recall': self.Recall,
            'f1': self.F1,
            'roc_auc': self.ROC_AUC
            }
        self.metric = self._metrics.get(metric, metric)
        self.best_score = 0.0
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        ans = 'MyLogReg class: '
        for key in self.__dict__:
            ans += key + '=' + str(self.__dict__[key]) + ', '
        return ans[:-2]
    
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose = False):
        X = X.copy()
        X.insert(0, 'intercept', 1)
        self.weights = pd.Series(1, index = X.columns)
        random.seed(self.random_state)
        for iter in range(1, self.n_iter + 1):
            y_pred = 1 / (1 + np.exp(-X.dot(self.weights))) #predict probabilities
            loss = -(y * np.log(y_pred + self.eps) + (1 - y) * np.log(1 - y_pred + self.eps)).sum() / len(y) #calculate loss
            if self.sgd_sample: #stochastic gradient descent
                sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample if isinstance(self.sgd_sample, int) else int(self.sgd_sample * X.shape[0]))
                X_sample = X.iloc[sample_rows_idx]
                y_sample = y.iloc[sample_rows_idx]
                y_pred_sample = 1 / (1 + np.exp(-X_sample.dot(self.weights)))
                gradient = (y_pred_sample - y_sample).dot(X_sample) / len(y_sample) #calculate gradient
            else: #classic gradient descent
                gradient = (y_pred - y).dot(X) / len(y) #calculate gradient
            if self.reg == 'l1' or self.reg == 'elasticnet': #l1 regularization (lasso)
                sgn = self.weights.copy()
                sgn[sgn > 0] = 1
                sgn[sgn < 0] = -1
                loss += self.l1_coef * self.weights.abs().sum()
                gradient += sgn * self.l1_coef
            if self.reg == 'l2' or self.reg == 'elasticnet': #l2 regularization (ridge)
                loss += self.l2_coef * (self.weights ** 2).sum()
                gradient += 2 * self.l2_coef * self.weights
            if callable(self.learning_rate): #update weights dynamically
                self.weights -= gradient * self.learning_rate(iter)
            else: #update weights statically
                self.weights -= gradient * self.learning_rate
            if verbose and iter % verbose == 0: #print log
                output = f'{iter} | loss: {loss}'
                if self.metric:
                    if self.metric != self.ROC_AUC:
                        y_pred[y_pred > 0.5] = 1
                        y_pred[y_pred <= 0.5] = 0
                        y_pred = y_pred.astype('int')
                    output += f' | {self.metric.__name__}: {self.metric(y, y_pred)}'
                if callable(self.learning_rate):
                    output += f' | learning rate: {self.learning_rate(iter)}'
                print(output)
        if self.metric: #calculate best score
            y_pred = 1 / (1 + np.exp(-X.dot(self.weights)))
            if self.metric != self.ROC_AUC:
                        y_pred[y_pred > 0.5] = 1
                        y_pred[y_pred <= 0.5] = 0
                        y_pred = y_pred.astype('int')
            self.best_score = self.metric(y, y_pred)

    def get_coef(self):
        return self.weights.to_numpy()[1:]
    
    def get_intercept(self):
        return self.weights.to_numpy()[0]
    
    def predict(self, X: pd.DataFrame):
        X = X.copy()
        X.insert(0, 'intercept', 1)
        y_pred = 1 / (1 + np.exp(-X.dot(self.weights)))
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        return y_pred.astype('int')

    def predict_proba(self, X: pd.DataFrame):
        X = X.copy()
        X.insert(0, 'intercept', 1)
        y_pred = 1 / (1 + np.exp(-X.dot(self.weights)))
        return y_pred
    
    def get_best_score(self):
        return self.best_score
    
    def Accuracy(self, y: pd.Series, y_pred: pd.Series):
        return (y == y_pred).sum() / len(y)
    
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
            criterion: str = 'entropy',
            samples: int = 0):
        self.column = column
        self.split = split
        self.ig = ig
        self.left = left
        self.right = right
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

class MyTreeClf():
    '''
        Decision tree classification
    '''

    def __init__(self, max_depth = 5, min_samples_split = 2, max_leafs = 20, bins = None, criterion = 'entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max(2, max_leafs)
        self.eps = 1e-15
        self.leafs_cnt = 0
        self.tree = Tree()
        self.bins = bins
        self.criterion = criterion
        self.total_samples = 0
        self.fi = {}

    def __str__(self):
        ans = 'MyTreeClf class:\n'
        for key in self.__dict__:
            prefix = '{\n' if isinstance(self.__dict__[key], Tree) else ''
            suffix = '\n},\n' if isinstance(self.__dict__[key], Tree) else ',\n'
            ans += f'{key}={prefix}{self.__dict__[key]}{suffix}'
        return ans[:-2]
    
    def print_tree(self):
        print(self.tree)
    
    def ShenonEntropy(self, classes: pd.Series):
        probas = classes.value_counts() / len(classes)
        probas = probas[probas > 0]
        return -(probas * np.log2(probas)).sum()
    
    def Gini(self, classes: pd.Series):
        probas = classes.value_counts() / len(classes)
        return 1 - (probas ** 2).sum()

    def InformationGain(self, X: pd.DataFrame, y: pd.Series, column: str, split: float):
        S0 = self.ShenonEntropy(y)
        left = y[X[column] <= split]
        right = y[X[column] > split]
        S1 = self.ShenonEntropy(left)
        S2 = self.ShenonEntropy(right)
        return S0 - (len(left) * S1 + len(right) * S2) / len(y)
    
    def GiniGain(self, X: pd.DataFrame, y: pd.Series, column: str, split: float):
        G0 = self.Gini(y)
        left = y[X[column] <= split]
        right = y[X[column] > split]
        G1 = self.Gini(left)
        G2 = self.Gini(right)
        return G0 - (len(left) * G1 + len(right) * G2) / len(y)
    
    def Gain(self, X: pd.DataFrame, y: pd.Series, column: str, split: float):
        return self.InformationGain(X, y, column, split) if self.criterion == 'entropy' else self.GiniGain(X, y, column, split) if self.criterion == 'gini' else float('-inf')

    def FeatureImportance(self, X: pd.DataFrame, y: pd.Series, column: str, split: float):
        return self.Gain(X, y, column, split) * len(y) / self.total_samples

    def get_best_split(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        ig = float('-inf')
        col_name = ''
        split_value = 0.0
        for column in X:
            splits = pd.Series()
            if self.bins:
                splits = self.hist[column]
            else:
                splits = np.convolve(np.sort(X[column].unique()), [0.5, 0.5], 'valid')
            for split in splits:
                tmp_ig = self.Gain(X, y, column, split) 
                if tmp_ig > ig:
                    ig = tmp_ig
                    col_name = column
                    split_value = split
        return col_name, split_value, ig
    
    def splits_count(self, X: pd.DataFrame) -> int:
        ans = 0
        for column in X:
            ans += len(self.hist[column][(self.hist[column] >= X[column].min()) & (self.hist[column] <= X[column].max())])
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
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.total_samples = len(X)
        for column in X:
            self.fi[column] = 0
        self.hist = pd.DataFrame()
        if self.bins:
            h = {}
            for column in X:
                native = np.convolve(np.sort(X[column].unique()), [0.5, 0.5], 'valid')
                if native.shape[0] <= self.bins - 1:
                    h[column] = native
                else:
                    _, bin_edges = np.histogram(X[column], self.bins)
                    h[column] = bin_edges[1:-1]
            self.hist = h
        self.tree = self.growth_tree(X, y, 0, 0)

    def predict(self, X: pd.DataFrame):
        ans = []
        for index, row in X.iterrows():
            ans += [self.tree.traversal(row)]
        return (pd.Series(ans) > 0.5).astype(int)
            
    def predict_proba(self, X: pd.DataFrame):
        ans = []
        for index, row in X.iterrows():
            ans += [self.tree.traversal(row)]
        return pd.Series(ans)
    
class MyKNNClf():
    
    def __init__(self, k: int = 3, metric: str = 'euclidean', weight: str = 'uniform') -> None:
        self.k: int = k
        self.train_size: tuple = (0, 0)
        self._metrics: dict = {'euclidean': self.Euclidean, 'chebyshev': self.Chebyshev, 'manhattan': self.Manhattan, 'cosine': self.Cosine}
        self.metric = self._metrics[metric]
        self.weight: str = weight
        
    def __str__(self) -> str:
        ans = 'MyKNNClf class:'
        for key in self.__dict__:
            ans += f' {key}={self.__dict__[key]},'
        return ans[:-1]
    
    def Euclidean(self, row: pd.Series, X: pd.DataFrame) -> pd.Series:
        return ((X - row) ** 2).sum(axis = 1) ** 0.5
    
    def Chebyshev(self, row: pd.Series, X: pd.DataFrame) -> pd.Series:
        return (X - row).abs().max(axis = 1)
    
    def Manhattan(self, row: pd.Series, X: pd.DataFrame) -> pd.Series:
        return (X - row).abs().sum(axis = 1)
    
    def Cosine(self, row: pd.Series, X: pd.DataFrame) -> pd.Series:
        return 1 - (X * row).sum(axis = 1) / (row ** 2).sum() ** 0.5 / (X ** 2).sum(axis = 1) ** 0.5
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X.reset_index(drop = True)
        self.y = y.reset_index(drop = True)
        self.train_size = X.shape
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = X.reset_index(drop = True)
        ans = []
        for index, row in X.iterrows():
            neighbors = self.metric(row, self.X).sort_values().head(self.k)
            indices = neighbors.index.tolist()
            an = []
            if self.weight == 'uniform':
                an = int(self.y.iloc[indices].mean() >= 0.5) 
            elif self.weight == 'rank':
                c = np.array(self.y.iloc[indices])
                r = np.array(range(1, self.k + 1))
                an = int((1 / r[np.where(c == 1)]).sum() / (1 / r).sum() >= 0.5)
            elif self.weight == 'distance':
                c = np.array(self.y.iloc[indices])
                d = np.array(neighbors)
                an = int((1 / d[np.where(c == 1)]).sum() / (1 / d).sum() >= 0.5)
            ans += [an]
        return pd.Series(ans)
    
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        X = X.reset_index(drop = True)
        ans = []
        for index, row in X.iterrows():
            neighbors = self.metric(row, self.X).sort_values().head(self.k)
            indices = neighbors.index.tolist()
            an = []
            if self.weight == 'uniform':
                an = self.y.iloc[indices].mean()
            elif self.weight == 'rank':
                c = np.array(self.y.iloc[indices])
                r = np.array(range(1, self.k + 1))
                an = (1 / r[np.where(c == 1)]).sum() / (1 / r).sum()
            elif self.weight == 'distance':
                c = np.array(self.y.iloc[indices])
                d = np.array(neighbors)
                an = (1 / d[np.where(c == 1)]).sum() / (1 / d).sum()
            ans += [an]
        return pd.Series(ans)

class MyBaggingClf():

    def __init__(self, estimator = None, n_estimators: int = 10, max_samples: float = 1.0, random_state = 42, oob_score = None) -> None:
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.oob_score_ = 0.0
        self._metrics = {'accuracy': self.Accuracy, 'precision': self.Precision, 'recall': self.Recall, 'f1': self.F1, 'roc_auc': self.ROC_AUC}
        self.metric = self._metrics[oob_score] if oob_score is not None else None 

    def __str__(self) -> str:
        ans = 'MyBaggingClf class: '
        for key in self.__dict__:
            ans += f'{key}={self.__dict__[key]}, '
        return ans[:-2]
    
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
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        X.reset_index(drop = True, inplace = True)
        y.reset_index(drop = True, inplace = True)
        random.seed(self.random_state)
        indexes = []
        for i in range(self.n_estimators):
            indexes += [random.choices(X.index, k = round(X.shape[0] * self.max_samples))]
        self.estimators = []
        oob_scores = []
        for i in range(self.n_estimators):
            X_s = X.loc[indexes[i]]
            y_s = y.loc[indexes[i]]
            estim = copy(self.estimator)
            estim.fit(X_s, y_s)
            self.estimators += [estim]
            oob_index = list(set(range(X.shape[0])) - set(indexes[i]))
            X_s_oob = X.loc[oob_index]
            #print(estim.predict_proba(X_s_oob))
            oob_scores += [pd.Series(np.array(estim.predict_proba(X_s_oob)), oob_index)]
        #print(pd.DataFrame(oob_scores).T.sort_index())
        oob_scores = (pd.DataFrame(oob_scores).T.sort_index().mean(axis = 1) > 0.5).astype(int)
        if self.metric is not None:
            self.oob_score_ = self.metric(y.iloc[oob_scores.index], oob_scores)
            
    def predict(self, X: pd.DataFrame, type: str):
        X.reset_index(drop = True, inplace = True)
        ans = []
        if type == 'mean':
            for estim in self.estimators:
                ans += [estim.predict_proba(X)]
            return (pd.DataFrame(ans).mean(axis = 0) > 0.5).astype(int)
        elif type == 'vote':
            for estim in self.estimators:
                ans += [estim.predict_proba(X)]
            return ((pd.DataFrame(ans) > 0.5).mean(axis = 0) >= 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame):
        X.reset_index(drop = True, inplace = True)
        ans = []
        for estim in self.estimators:
                ans += [estim.predict_proba(X)]
        return pd.DataFrame(ans).mean(axis = 0)
                
                
X, y = make_classification(n_samples = 100, n_features=4, n_informative=2, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

a = MyBaggingClf(MyKNNClf(), max_samples = 0.2, n_estimators = 5)
a.fit(X, y)
print(a.oob_score_)
        
    
