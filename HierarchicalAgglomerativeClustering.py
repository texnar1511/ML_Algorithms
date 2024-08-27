import numpy as np
import pandas as pd

class MyAgglomerative():
    
    def __init__(self, n_clusters: int = 3, metric = 'euclidean') -> None:
        self.n_clusters = n_clusters
        self._metrics = {'euclidean': self.Euclidean, 'chebyshev': self.Chebyshev, 'manhattan': self.Manhattan, 'cosine': self.Cosine}
        self.metric = self._metrics[metric]
        
    def __str__(self) -> str:
        ans = 'MyAgglomerative class:'
        for key in self.__dict__:
            ans += f' {key}={self.__dict__[key]},'
        return ans[:-1]
    
    def Euclidean(self, x: pd.Series, y: pd.Series) -> float:
        return ((x - y) ** 2).sum() ** 0.5
    
    def Chebyshev(self, x: pd.Series, y: pd.Series) -> float:
        return (x - y).abs().max()

    def Manhattan(self, x: pd.Series, y: pd.Series) -> float:
        return (x - y).abs().sum()
    
    def Cosine(self, x: pd.Series, y: pd.Series) -> float:
        return 1 - (x * y).sum() / (x ** 2).sum() ** 0.5 / (y ** 2).sum() ** 0.5
    
    def getUnion(self, X: pd.DataFrame, clusters: list):
        ans = -1, -1, float('inf')
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster1 = clusters[i]
                cluster2 = clusters[j]
                envoy1 = X.iloc[cluster1] if type(cluster1) == int else X.iloc[cluster1].mean(axis = 0)
                envoy2 = X.iloc[cluster2] if type(cluster2) == int else X.iloc[cluster2].mean(axis = 0)
                distance = self.metric(envoy1, envoy2)
                if distance < ans[2]:
                    ans = cluster1, cluster2, distance
        return ans
                
    def fit_predict(self, X: pd.DataFrame) -> list:
        clusters = X.index.tolist()
        #print(clusters)
        while len(clusters) > self.n_clusters:
            res = self.getUnion(X, clusters)
            #print(res)
            clusters.remove(res[0])
            clusters.remove(res[1])
            clusters += [([res[0]] if type(res[0]) == int else res[0]) + ([res[1]] if type(res[1]) == int else res[1])]
            #print(clusters)
        ans = [-1] * len(X)
        for i in range(len(clusters)):
            if type(clusters[i]) == int:
                ans[clusters[i]] = i
            else:
                for j in clusters[i]:
                    ans[j] = i
        return ans
    

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples = 100, centers = 3, n_features = 2, random_state = 0)

X = pd.DataFrame(X)
X.columns = [f'col_{i}' for i in X.columns]

cl = MyAgglomerative(3)
print(cl.fit_predict(X))