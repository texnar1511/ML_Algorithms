import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

class MyPCA():
    
    def __init__(self, n_components = 3):
        self.n_components = n_components
    
    def __str__(self):
        ans = 'MyPCA class:'
        for key in self.__dict__:
            ans += f' {key}={self.__dict__[key]},'
        return ans[:-1]
    
    def fit_transform(self, X: pd.DataFrame):
        X = X - X.mean(axis = 0)
        cov = X.cov()
        eigh = np.linalg.eigh(cov)
        W_pca = eigh[1].T[::-1][:self.n_components]
        return np.dot(X, W_pca.T)

X, y = make_classification(n_samples = 10, n_features = 5, n_informative = 2, random_state = 42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]
    
a = MyPCA(3)
print(a)
b = a.fit_transform(X)
print(b)
