import numpy as np
import pandas as pd
import random
import sklearn.metrics as metrics

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

    def __init__(self, max_depth = 5, min_samples_split = 2, max_leafs = 20, bins = None, criterion = 'entropy', total_samples = 1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max(2, max_leafs)
        self.eps = 1e-15
        self.leafs_cnt = 0
        self.tree = Tree()
        self.bins = bins
        self.criterion = criterion
        self.total_samples = total_samples
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
            node = Tree(kind = 'leaf', proba = y.mean(), samples = len(y))
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
        #self.total_samples = len(X)
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

    def predict(self, X: pd.DataFrame) -> pd.Series:
        ans = []
        for index, row in X.iterrows():
            ans += [self.tree.traversal(row)]
        return (pd.Series(ans) > 0.5).astype(int)
            
    def predict_proba(self, X: pd.DataFrame):
        ans = []
        for index, row in X.iterrows():
            ans += [self.tree.traversal(row)]
        return pd.Series(ans)

class MyForestClf():

    def __init__(self, 
                 n_estimators = 10, 
                 max_features = 0.5, 
                 max_samples = 0.5, 
                 max_depth = 5,
                 min_samples_split = 2,
                 max_leafs = 20,
                 bins = 16,
                 criterion = 'entropy',
                 oob_score = None,
                 random_state = 42
                 ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.criterion = criterion
        self.random_state = random_state
        self.leafs_cnt = 0
        self.forest: list[MyTreeClf] = []
        self.fi = {}
        self.oob_score = oob_score
        self.oob_score_ = 0
        self.metrics = {'accuracy': self.Accuracy, 'precision': self.Precision, 'recall': self.Recall, 'f1': self.F1, 'roc_auc': self.ROC_AUC}


    def __str__(self):
        ans = 'MyForestClf class:'
        for key in self.__dict__:
            ans += f' {key}={self.__dict__[key]},'
        return ans[:-1]
    
    def Accuracy(self, y: pd.Series, y_pred: pd.Series):
        return (y == y_pred).mean()
    
    def Precision(self, y: pd.Series, y_pred: pd.Series):
        return (y * y_pred).sum() / y_pred.sum()
    
    def Recall(self, y: pd.Series, y_pred: pd.Series):
        return (y * y_pred).sum() / y.sum()
    
    def F1(self, y: pd.Series, y_pred: pd.Series):
        return 2 * (y * y_pred).sum() / (y.sum() + y_pred.sum())
    
    def ROC_AUC(self, y: pd.Series, y_pred_proba: pd.Series):
        y_pred_proba = y_pred_proba.round(10)#.reset_index(drop = True)
        #y = y.reset_index(drop = True)
        #print(y)
        #print(y_pred_proba)
        df = pd.concat([y_pred_proba, y], axis = 1)
        df.columns = [0, 1]
        #print(df)
        #print(df.columns)
        df = df.sort_values([0, 1], ascending = [False, False]).reset_index()
        roc_auc = 0.0
        for i in df.index:
            if df.iloc[i, 2] == 0:
                roc_auc += df.iloc[:i, 2][df.iloc[:i, 1] > df.iloc[i, 1]].sum()
                roc_auc += df.iloc[:i, 2][df.iloc[:i, 1] == df.iloc[i, 1]].sum() / 2
        s = y.sum()
        return roc_auc / s / (len(y) - s)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        random.seed(self.random_state)
        for column in X:
            self.fi[column] = 0
        preds_oob = []
        for i in range(self.n_estimators):
            cols_idx = random.sample(list(X.columns), round(X.shape[1] * self.max_features))
            rows_idx = random.sample(range(X.shape[0]), round(X.shape[0] * self.max_samples))
            X_s = X.loc[rows_idx, cols_idx]
            y_s = y.loc[rows_idx]
            tree = MyTreeClf(self.max_depth, self.min_samples_split, self.max_leafs, self.bins, self.criterion, len(X))
            tree.fit(X_s, y_s)
            self.forest += [tree]
            self.leafs_cnt += tree.leafs_cnt
            rows_idx_oob = list(set(range(X.shape[0])) - set(rows_idx))
            X_s_oob = X.loc[rows_idx_oob, cols_idx]
            pred_oob = tree.predict_proba(X_s_oob)
            pred_oob.index = X_s_oob.index
            preds_oob += [pred_oob]
        for column in X:
            self.fi[column] = sum([tree.fi.get(column, 0) for tree in self.forest])
        preds_oob = pd.DataFrame(preds_oob).mean(axis = 0)
        #print(preds_oob)
        #print(y.iloc[preds_oob.index])
        #print(metrics.roc_auc_score(y.iloc[preds_oob.index], preds_oob))
        if self.oob_score:
            if self.oob_score != 'roc_auc':
                preds_oob = (preds_oob > 0.5).astype(int)
            self.oob_score_ = self.metrics[self.oob_score](y.iloc[preds_oob.index], preds_oob)

    def predict(self, X: pd.DataFrame, type: str) -> pd.Series:
        if type == 'vote':
            return (pd.DataFrame([tree.predict(X) for tree in self.forest]).mean(axis = 0) >= 0.5).astype(int)
        elif type == 'mean':
            return (pd.DataFrame([tree.predict_proba(X) for tree in self.forest]).mean(axis = 0) > 0.5).astype(int)
        
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        return pd.DataFrame([tree.predict_proba(X) for tree in self.forest]).mean(axis = 0)

df = pd.read_csv('banknote+authentication.zip', header = None)
df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
X, y = df.iloc[:,:4], df['target']

a = MyForestClf(oob_score = 'accuracy')
a.fit(X, y)
print(a.oob_score_)
#print(a.predict_proba(X))