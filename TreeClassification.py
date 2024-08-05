import pandas as pd
from sklearn.datasets import make_classification
import numpy as np

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
        self.fi = {}
        self.total_samples = 0

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
    



#d = {
#    'max_depth': 15,
#    'min_samples_split': 20,
#    'max_leafs': 30,
#    'bins': 6
#    }

#X, y = make_classification(n_samples=150, n_features=5, n_informative=3, random_state=42)
#X = pd.DataFrame(X).round(2)
#y = pd.Series(y)
#X.columns = [f'col_{col}' for col in X.columns]
#test = X.sample(20, random_state=42)

d = [{
    'max_depth': 1,
    'min_samples_split': 1,
    'max_leafs': 2,
    'bins': 8,
    'criterion': 'entropy'
    },
    {
    'max_depth': 3,
    'min_samples_split': 2,
    'max_leafs': 5,
    'bins': None,
    'criterion': 'entropy'
    },
    {
    'max_depth': 5,
    'min_samples_split': 200,
    'max_leafs': 10,
    'bins': 4,
    'criterion': 'entropy'
    },
    {
    'max_depth': 4,
    'min_samples_split': 100,
    'max_leafs': 17,
    'bins': 16,
    'criterion': 'entropy'
    },
    {
    'max_depth': 10,
    'min_samples_split': 40,
    'max_leafs': 21,
    'bins': 10,
    'criterion': 'entropy'
    },
    {
    'max_depth': 15,
    'min_samples_split': 20,
    'max_leafs': 30,
    'bins': 6,
    'criterion': 'entropy'
    },
]

df = pd.read_csv('banknote+authentication.zip', header=None)
df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
X, y = df.iloc[:,:4], df['target']

#X, y = make_classification(n_samples=150, n_features=5, n_informative=3, random_state=42)
#X = pd.DataFrame(X).round(2)
#y = pd.Series(y)
#X.columns = [f'col_{col}' for col in X.columns]

for i in d:
    a = MyTreeClf(**i)
    a.fit(X, y)
    print(a.leafs_cnt)
    print(a.tree.proba_sum())
    

#print(a.predict_proba(test).sum())
#print(a.predict_proba(X).sum())
#print(a.fi)

#d = [{
#    'max_depth': 8,
#    'min_samples_split': 5,
#    'max_leafs': 15,
#    'bins': 10
#    },
#    {
#    'max_depth': 5,
#    'min_samples_split': 5,
#    'max_leafs': 10,
#    'bins': 15
#    },
#]
#
#X, y = make_classification(n_samples=150, n_features=5, n_informative=3, random_state=42)
#X = pd.DataFrame(X).round(2)
#y = pd.Series(y)
#X.columns = [f'col_{col}' for col in X.columns]
#
#for i in d:
#    a = MyTreeClf(**i, criterion = 'entropy')
#    a.fit(X, y)
#    a.print_tree()
#    print(a.leafs_cnt)
#    print(a.tree.proba_sum())
#
