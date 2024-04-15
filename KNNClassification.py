import pandas as pd
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=14, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

X_test, y_test = make_classification(n_samples=100, n_features=14, n_informative=10, random_state=42)
X_test = pd.DataFrame(X_test)
y_test = pd.Series(y)
X_test.columns = [f'col_{col}' for col in X_test.columns]

class MyKNNClf:

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
        ans = 'MyKNNClf class: '
        for key in self.__dict__:
            ans += f'{key}={self.__dict__[key]}, '
        return ans[:-2]
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X_train = X
        self.y_train = y
        self.train_size = X.shape

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        ans = []
        for i in range(X_test.shape[0]):
            Dist = self.metric(self.X_train, X_test.iloc[i]).sort_values().iloc[:self.k]
            Dist.name = 'distance'
            Classes = self.y_train.iloc[Dist.index]
            Classes.name = 'class'
            Rank = pd.Series(range(1, self.k + 1), name = 'rank', index = Dist.index)
            DCR = pd.concat([Dist, Classes, Rank], axis = 1)
            DCR['distance'] = 1 / DCR['distance']
            DCR['rank'] = 1 / DCR['rank']
            value = 0
            if self.weight == 'uniform':
                value = 1 if 2 * Classes.sum() >= len(Classes) else 0
            elif self.weight == 'distance':
                value = 1 if 2 * DCR[DCR['class'] == 1]['distance'].sum() / DCR['distance'].sum() >= 1 else 0
            elif self.weight == 'rank':
                value = 1 if 2 * DCR[DCR['class'] == 1]['rank'].sum() / DCR['rank'].sum() >= 1 else 0
            else:
                pass
            ans += [value]
        return pd.Series(ans)
    
    def predict_proba(self, X_test: pd.DataFrame) -> pd.Series:
        ans = []
        for i in range(X_test.shape[0]):
            Dist = self.metric(self.X_train, X_test.iloc[i]).sort_values().iloc[:self.k]
            Dist.name = 'distance'
            Classes = self.y_train.iloc[Dist.index]
            Classes.name = 'class'
            Rank = pd.Series(range(1, self.k + 1), name = 'rank', index = Dist.index)
            DCR = pd.concat([Dist, Classes, Rank], axis = 1)
            DCR['distance'] = 1 / DCR['distance']
            DCR['rank'] = 1 / DCR['rank']
            value = 0.0
            if self.weight == 'uniform':
                value = Classes.sum() / len(Classes)
            elif self.weight == 'distance':
                value = DCR[DCR['class'] == 1]['distance'].sum() / DCR['distance'].sum()
            elif self.weight == 'rank':
                value = DCR[DCR['class'] == 1]['rank'].sum() / DCR['rank'].sum()
            else:
                pass
            ans += [value]
        return pd.Series(ans)
    
    def Euclidean(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        return ((X - y) ** 2).sum(axis = 1) ** (1 / 2)
    
    def Chebyshev(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        return (X - y).abs().max(axis = 1)
    
    def Manhattan(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        return (X - y).abs().sum(axis = 1)
    
    def Cosine(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        return 1 - (X * y).sum(axis = 1) / (y ** 2).sum() ** (1 / 2) / (X ** 2).sum(axis = 1) ** (1 / 2)


        
    
#a = MyKNNClf(metric = 'euclidean', weight = 'rank')
#print(a)

#a.fit(X, y)
#print(a.predict(X_test).sum())
#print(a.predict_proba(X_test).sum())

