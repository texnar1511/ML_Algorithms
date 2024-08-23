import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class MyKMeans():

    def __init__(self, n_clusters = 3, max_iter = 10, n_init = 3, random_state = 42) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.eps = 1e-12

    def __str__(self) -> str:
        ans = 'MyKMeans class: '
        for key in self.__dict__:
            ans += f'{key}={self.__dict__[key]}, '
        return ans[:-2]
    
    def Euclidean(self, x: pd.Series, y: pd.Series) -> float:
        return ((x - y) ** 2).sum() ** 0.5
    
    def check(self, centers, new_centers):
        centers = np.array(centers)
        new_centers = np.array(new_centers)
        #print(centers)
        #print(new_centers)
        return (centers == new_centers).all()
    
    def WCSS(self, X: pd.DataFrame, centers: list) -> float:
        ans = 0.0
        for index, row in X.iterrows():
            ans += min([self.Euclidean(row, center) ** 2 for center in centers])
        return ans
    
    def fit(self, X: pd.DataFrame):
        np.random.seed(self.random_state)
        centroids = []
        for it in range(self.n_init):
            centers = []
            for i in range(self.n_clusters):
                center = []
                for j in X.columns:
                    center += [np.random.uniform(X[j].min(), X[j].max())]
                centers += [center]
            ii = 0
            while ii < self.max_iter:
                clusters = {}
                for i in range(self.n_clusters):
                    clusters[i] = []
                for index, row in X.iterrows():
                    clusters[np.argmin([self.Euclidean(row, center) for center in centers])] += [index]
                new_centers = [-1] * self.n_clusters
                for k, v in clusters.items():
                    if v:
                        new_centers[k] = X.iloc[v].mean(axis = 0).tolist()
                    else:
                        new_centers[k] = centers[k]
                if self.check(centers, new_centers):
                    centers = new_centers
                    break
                centers = new_centers
                ii += 1
            centroids += [centers]
        wcss = [self.WCSS(X, centers) for centers in centroids]
        self.cluster_centers_ = centroids[np.argmin(wcss)]
        self.inertia_ = np.min(wcss)
        
    def predict(self, X: pd.DataFrame) -> list:
        ans = []
        for index, row in X.iterrows():
            ans += [np.argmin([self.Euclidean(row, center) for center in self.cluster_centers_])]
        return ans
        
            

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples = 100, centers = 3, n_features = 2, random_state = 0)

X = pd.DataFrame(X)
X.columns = [f'col_{i}' for i in X.columns]
#print(X)

#with open('ML_Algorithms/test.npy', 'rb') as f:
#    X = np.load(f)
    
#X = pd.DataFrame(X)
#X.columns = [f'col_{i}' for i in X.columns]

a = MyKMeans(3)
print(a.fit(X))
print(a.predict(X))
#print(a.inertia_)
#print(np.sum(a.cluster_centers_))
#print(np.array(a.cluster_centers_))
#print(a.predict(X))
#print(y)

#plt.scatter(X.loc[:, 0], X.loc[:, 1], c = y)
#plt.scatter(np.array(a.cluster_centers_)[:, 0], np.array(a.cluster_centers_)[:, 1], c = 'orange')
#plt.show()