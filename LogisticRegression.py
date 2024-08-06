import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import random

X, y = make_classification(n_samples=1000, n_features=14, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

class MyLogReg:
    '''
    Logistic regression
    '''

    def __init__(
            self, 
            n_iter: int = 10, 
            learning_rate: float = 0.1, 
            weights = None, 
            metric = None, 
            reg = None, 
            l1_coef: float = 0.0, 
            l2_coef: float = 0.0, 
            sgd_sample = None, 
            random_state = 42
            ):
        '''
        n_iter: int, default = 10 
            Number of Gradient Descending algorithm steps
        learning_rate: float de
        '''
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.eps = 1e-15
        self._metrics = {
            'accuracy': self.Accuracy, 
            'precision': self.Precision, 
            'recall': self.Recall,
            'f1': self.F1,
            'roc_auc': self.ROC_AUC
            }
        self.metric = self._metrics.get(metric, metric)
        self.best_score = 0.0
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        ans = 'MyLogReg class: '
        for key in self.__dict__:
            ans += key + '=' + str(self.__dict__[key]) + ', '
        return ans[:-2]
    
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose = False):
        X = X.copy()
        X.insert(0, 'intercept', 1)
        self.weights = pd.Series(1, index = X.columns)
        random.seed(self.random_state)
        for iter in range(1, self.n_iter + 1):
            y_pred = 1 / (1 + np.exp(-X.dot(self.weights))) #predict probabilities
            loss = -(y * np.log(y_pred + self.eps) + (1 - y) * np.log(1 - y_pred + self.eps)).sum() / len(y) #calculate loss
            if self.sgd_sample: #stochastic gradient descent
                sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample if isinstance(self.sgd_sample, int) else int(self.sgd_sample * X.shape[0]))
                X_sample = X.iloc[sample_rows_idx]
                y_sample = y.iloc[sample_rows_idx]
                y_pred_sample = 1 / (1 + np.exp(-X_sample.dot(self.weights)))
                gradient = (y_pred_sample - y_sample).dot(X_sample) / len(y_sample) #calculate gradient
            else: #classic gradient descent
                gradient = (y_pred - y).dot(X) / len(y) #calculate gradient
            if self.reg == 'l1' or self.reg == 'elasticnet': #l1 regularization (lasso)
                sgn = self.weights.copy()
                sgn[sgn > 0] = 1
                sgn[sgn < 0] = -1
                loss += self.l1_coef * self.weights.abs().sum()
                gradient += sgn * self.l1_coef
            if self.reg == 'l2' or self.reg == 'elasticnet': #l2 regularization (ridge)
                loss += self.l2_coef * (self.weights ** 2).sum()
                gradient += 2 * self.l2_coef * self.weights
            if callable(self.learning_rate): #update weights dynamically
                self.weights -= gradient * self.learning_rate(iter)
            else: #update weights statically
                self.weights -= gradient * self.learning_rate
            if verbose and iter % verbose == 0: #print log
                output = f'{iter} | loss: {loss}'
                if self.metric:
                    if self.metric != self.ROC_AUC:
                        y_pred[y_pred > 0.5] = 1
                        y_pred[y_pred <= 0.5] = 0
                        y_pred = y_pred.astype('int')
                    output += f' | {self.metric.__name__}: {self.metric(y, y_pred)}'
                if callable(self.learning_rate):
                    output += f' | learning rate: {self.learning_rate(iter)}'
                print(output)
        if self.metric: #calculate best score
            y_pred = 1 / (1 + np.exp(-X.dot(self.weights)))
            if self.metric != self.ROC_AUC:
                        y_pred[y_pred > 0.5] = 1
                        y_pred[y_pred <= 0.5] = 0
                        y_pred = y_pred.astype('int')
            self.best_score = self.metric(y, y_pred)

    def get_coef(self):
        return self.weights.to_numpy()[1:]
    
    def get_intercept(self):
        return self.weights.to_numpy()[0]
    
    def predict(self, X: pd.DataFrame):
        X = X.copy()
        X.insert(0, 'intercept', 1)
        y_pred = 1 / (1 + np.exp(-X.dot(self.weights)))
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        return y_pred.astype('int')

    def predict_proba(self, X: pd.DataFrame):
        X = X.copy()
        X.insert(0, 'intercept', 1)
        y_pred = 1 / (1 + np.exp(-X.dot(self.weights)))
        return y_pred
    
    def get_best_score(self):
        return self.best_score
    
    def Accuracy(self, y: pd.Series, y_pred: pd.Series):
        return (y == y_pred).sum() / len(y)
    
    def Precision(self, y: pd.Series, y_pred: pd.Series):
        return (y * y_pred).sum() / y_pred.sum()
    
    def Recall(self, y: pd.Series, y_pred: pd.Series):
        return (y * y_pred).sum() / y.sum()
    
    def F1(self, y: pd.Series, y_pred: pd.Series):
        return 2 * (y * y_pred).sum() / (y.sum() + y_pred.sum())
    
    def ROC_AUC(self, y: pd.Series, y_pred_proba: pd.Series):
        y_pred_proba = y_pred_proba.round(10)
        df = pd.concat([y_pred_proba, y], axis = 1)
        df.columns = [0, 1]
        df = df.sort_values([0, 1], ascending = [False, False]).reset_index()
        roc_auc = 0.0
        for i in df.index:
            if df.iloc[i, 2] == 0:
                roc_auc += df.iloc[:i, 2][df.iloc[:i, 1] > df.iloc[i, 1]].sum()
                roc_auc += df.iloc[:i, 2][df.iloc[:i, 1] == df.iloc[i, 1]].sum() / 2
        s = y.sum()
        return roc_auc / s / (len(y) - s)
        


a = MyLogReg(learning_rate = lambda iter: 0.5 * (0.85 ** iter), metric = 'roc_auc', reg = 'elasticnet', l1_coef = 0.5, l2_coef = 0.5, sgd_sample = 100)
a.fit(X, y, verbose = 1)
#print(a.get_coef())
#print(a.get_intercept())
#print(a.predict(X))
#print(a.predict_proba(X))
y_pred = a.predict(X)
y_pred_proba = a.predict_proba(X)
print(a.get_best_score())
#print(y_pred_proba)
#print(y)
#print(y.sum())
#print(len(y) - y.sum())
#print(a.ROC_AUC(y, y_pred_proba))
#print(y_pred_proba)
#print(y.iloc[:-1])
#print(y.iloc[:0].sum())

#print(y_pred_proba.sort_values(ascending = False))
    
