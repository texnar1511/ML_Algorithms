import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

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
        self.X = X
        self.y = y
        self.train_size = X.shape
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
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
    
    
X, y = make_classification(n_samples=15, n_features=14, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

X_test, y_test = make_classification(n_samples=15, n_features=14, n_informative=10, random_state=40)
X_test = pd.DataFrame(X_test)
y_test = pd.Series(y_test)
X_test.columns = [f'col_{col}' for col in X_test.columns]

a = MyKNNClf(weight = 'distance')
a.fit(X, y)
print(a.predict(X_test))
print(a.predict_proba(X_test))
