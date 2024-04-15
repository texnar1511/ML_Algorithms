import pandas as pd
from sklearn.datasets import make_regression
import random

X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

class MyKNNReg():
    '''
        K Nearest Neighbors Regression
    '''

    def __init__(self, k = 3, metric = 'euclidean', weight = 'uniform'):
        self.k = k
        self.train_size = (0, 0)
        self._metrics = {
            'euclidean': self.Euclidean,
            'chebyshev': self.Chebyshev,
            'manhattan': self.Manhattan,
            'cosine': self.Cosine
            }
        self.metric = self._metrics.get(metric, None)
        self.weight = weight

    def __str__(self):
        ans = 'MyKNNReg class: '
        for key in self.__dict__:
            ans += f'{key}={self.__dict__[key]}, '
        return ans[:-2]
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X_train = X
        self.y_train = y
        self.train_size = X.shape

    def Euclidean(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        return ((X - y) ** 2).sum(axis = 1) ** (1 / 2)
    
    def Chebyshev(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        return (X - y).abs().max(axis = 1)
    
    def Manhattan(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        return (X - y).abs().sum(axis = 1)
    
    def Cosine(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        return 1 - (X * y).sum(axis = 1) / (y ** 2).sum() ** (1 / 2) / (X ** 2).sum(axis = 1) ** (1 / 2)

    def predict(self, X_test: pd.DataFrame):
        ans = []
        for i in range(X_test.shape[0]):
            distances = self.metric(self.X_train, X_test.iloc[i]).sort_values().iloc[:self.k]
            dependencies = self.y_train.iloc[distances.index]
            if self.weight == 'rank':
                dependencies = dependencies / range(1, self.k + 1)
                avg = dependencies.sum() / sum(1 / x for x in range(1, self.k + 1))
            elif self.weight == 'distance':
                dependencies = dependencies / distances
                avg = dependencies.sum() / (1 / distances).sum()
            else:
                avg = dependencies.mean()
            ans += [avg]
        return pd.Series(ans)

    
a = MyKNNReg(k = 5, metric = 'cosine', weight = 'rank')
print(a)
a.fit(X, y)
print(a.predict(X))
