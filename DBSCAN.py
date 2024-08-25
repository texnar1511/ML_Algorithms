import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable

class MyDBSCAN():
    
    def __init__(self, eps: float = 3, min_samples: int = 3, metric: str = 'euclidean') -> None:
        self.eps = eps
        self.min_samples = min_samples
        self._metrics: dict[str, Callable[[pd.DataFrame, pd.Series], pd.Series]] = {'euclidean': self.Euclidean, 'chebyshev': self.Chebyshev, 'manhattan': self.Manhattan, 'cosine': self.Cosine}
        self.metric: Callable[[pd.DataFrame, pd.Series], pd.Series] = self._metrics.get(metric, metric)
        
    def __str__(self) -> str:
        ans = 'MyDBSCAN class: '
        for i in self.__dict__:
            ans += f'{i}={self.__dict__[i]}, '
        return ans[:-2]
    
    def Euclidean(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        return ((X - y) ** 2).sum(axis = 1) ** 0.5
    
    def Chebyshev(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        return (X - y).abs().max(axis = 1)
    
    def Manhattan(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        return (X - y).abs().sum(axis = 1)
    
    def Cosine(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        return 1 - (X * y).sum(axis = 1) / (X ** 2).sum(axis = 1) ** 0.5 / (y ** 2).sum() ** 0.5
    
    def findNeighbors(self, X: pd.DataFrame, index: int, row: pd.Series) -> list:
        neigh = self.metric(X, row) < self.eps
        ans = neigh.index[neigh].tolist()
        ans.remove(index)
        return ans
    
    def classifyPoints(self, X: pd.DataFrame):
        cores = []
        borders = []
        outliers = []
        for index, row in X.iterrows():
            if len(self.findNeighbors(X, index, row)) >= self.min_samples:
                cores += [index]
        for index, row in X.iterrows():
            if index not in cores:
                neigh = self.findNeighbors(X, index, row)
                if len(set(neigh) & set(cores)):
                    borders += [index]
                else:
                    outliers += [index]
        return cores, borders, outliers
    
    def fit_predict(self, X: pd.DataFrame):
        clusters = [-1] * len(X)
        cores, borders, outliers = self.classifyPoints(X)
        for index, row in X.iterrows():
            if index in cores and clusters[index] == -1:
                    new_clust = max(clusters) + 1
                    clusters[index] = new_clust
                    neigh = self.findNeighbors(X, index, row)
                    i = 0
                    while i < len(neigh):
                        if neigh[i] in borders:
                            clusters[neigh[i]] = new_clust
                        elif neigh[i] in cores:
                            clusters[neigh[i]] = new_clust
                            neigh += [n for n in self.findNeighbors(X, neigh[i], X.loc[neigh[i]]) if n not in neigh and clusters[n] == -1]
                        i += 1                
        return clusters
        

with open('ML_Algorithms/test.npy', 'rb') as f:
    a = np.load(f)
    
from sklearn.datasets import make_blobs    
    
a, _ = make_blobs(n_samples = 1000, centers = 5, n_features = 2, cluster_std = 0.6, random_state = 0)

X = pd.DataFrame(a)
print(X.shape)

clust = MyDBSCAN(0.5, 5, 'euclidean')

from collections import Counter

#print(clust)
cl = clust.fit_predict(X)
print(cl)
print(len(cl))
print(len(X))

plt.scatter(a[:, 0], a[:, 1], c = cl)
plt.show()
