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
    
    def fill_missing_rows(self, new_centers: pd.DataFrame, centers: pd.DataFrame) -> pd.DataFrame:
        for i in set(centers.index) - set(new_centers.index):
            new_centers.loc[i] = centers.loc[i]
        return new_centers

    
    def fit_clusters(self, X: pd.DataFrame) -> pd.DataFrame:
        centers = pd.DataFrame(np.random.uniform(X.min(axis = 0), X.max(axis = 0), (self.n_clusters, X.shape[1])))
        for i in range(self.max_iter):
            distances = cdist(X, centers)
            new_centers = X.groupby(distances.argmin(axis = 1)).mean()
            #new_centers = self.fill_missing_rows(new_centers, centers)
            if (((new_centers - centers) ** 2).sum(axis = 1) ** (1 / 2)).sum() < self.eps:
                return new_centers
            centers = new_centers
        return centers
    
    def WCSS(self, X: pd.DataFrame, centers: pd.DataFrame):
        distances = cdist(X, centers)
        gb = X.groupby(distances.argmin(axis = 1))
        wcss = []
        for key, item in gb:
            wcss += [((item - centers.loc[key]) ** 2).sum().sum()]
        return sum(wcss)

    
    def fit(self, X: pd.DataFrame):
        np.random.seed(seed = self.random_state)
        group_centers = []
        group_wcss = []
        for i in range(self.n_init):
            group_centers += [self.fit_clusters(X)]
            group_wcss += [self.WCSS(X, group_centers[i])]
        self.inertia_ = min(group_wcss)
        self.cluster_centers_ = group_centers[np.argmin(group_wcss)].values.tolist()

    def predict(self, X: pd.DataFrame):
        return cdist(X, self.cluster_centers_).argmin(axis = 1)
            

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples = 10, centers = 3, n_features = 2, random_state = 0)

X = pd.DataFrame(X)
print(X)

a = MyKMeans(3, 10, 3)
a.fit(X)
print(a.inertia_)
print(np.sum(a.cluster_centers_))
print(np.array(a.cluster_centers_))
print(a.predict(X))
print(y)

#plt.scatter(X.loc[:, 0], X.loc[:, 1], c = y)
#plt.scatter(np.array(a.cluster_centers_)[:, 0], np.array(a.cluster_centers_)[:, 1], c = 'orange')
#plt.show()