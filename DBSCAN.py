import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MyDBSCAN():
    
    def __init__(self, eps: float = 3, min_samples: int = 3, metric: str = 'euclidean') -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self._metrics = {'euclidean': self.Euclidean, 'chebyshev': self.Chebyshev, 'manhattan': self.Manhattan, 'cosine': self.Cosine}
        
    def __str__(self) -> str:
        ans = 'MyDBSCAN class: '
        for i in self.__dict__:
            ans += f'{i}={self.__dict__[i]}, '
        return ans[:-2]
    
    def Euclidean(self, X: pd.DataFrame, y: pd.Series):
        return ((X - y) ** 2).sum(axis = 1) ** 0.5
    
    def Chebyshev(self, X: pd.DataFrame, y: pd.Series):
        return (X - y).abs().max(axis = 1)
    
    def Manhattan(self, X: pd.DataFrame, y: pd.Series):
        return (X - y).abs().sum(axis = 1)
    
    def Cosine(self, X: pd.DataFrame, y: pd.Series):
        return 1 - (X * y).sum(axis = 1) / (X ** 2).sum(axis = 1) ** 0.5 / (y ** 2).sum() ** 0.5
    
    def fit_predict(self, X: pd.DataFrame):
        cores = []
        borders = []
        outliers = []
        clusters = {-1: []}
        for index, row in X.iterrows():
            if index not in cores and index not in borders and index not in outliers:
                distances = self._metrics[self.metric](X, row)
                neighbors_mask = distances < self.eps
                if neighbors_mask.sum() >= self.min_samples + 1:
                    cores += [index]
                    cluster_idx = max(clusters.keys()) + 1
                    clusters[cluster_idx] = [index]
                    queue = [(id, it) for id, it in X[neighbors_mask].iterrows() if id not in cores and id not in borders]
                    while queue:
                        idx, item = queue.pop(0)
                        item_dist = self._metrics[self.metric](X, item)
                        item_mask = item_dist < self.eps
                        if idx in outliers or item_mask.sum() < self.min_samples + 1:
                            borders += [idx]
                            if idx in outliers:
                                outliers.remove(idx)
                        if idx not in cores and item_mask.sum() >= self.min_samples + 1:
                            cores += [idx]
                            clusters[cluster_idx] += [idx]
                            queue += [(id, it) for id, it in X[item_mask].iterrows() if id not in cores and id not in borders and id not in [ides for ides, itemes in queue]]
                else:
                    outliers += [index]
        clusters[-1] = outliers
        ans = [0] * len(X)
        for k, v in clusters.items():
            for i in v:
                ans[i] = k
        return ans

with open('ML_Algorithms/test.npy', 'rb') as f:
    a = np.load(f)

X = pd.DataFrame(a)
print(X.shape)

clust = MyDBSCAN(0.3, 3, 'euclidean')

from collections import Counter

print(clust)
cl = clust.fit_predict(X)
print(Counter(cl))

plt.scatter(a[:, 0], a[:, 1], c = cl)
plt.show()
