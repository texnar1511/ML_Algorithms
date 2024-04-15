import pandas as pd
from sklearn.datasets import make_regression
import random

X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

X_test, y_test = make_regression(n_samples=10, n_features=14, n_informative=10, noise=15, random_state=42)
X_test = pd.DataFrame(X_test)
X_test.columns = [f'col_{col}' for col in X_test.columns]

class MyLineReg():

    def __init__(self, n_iter = 100, learning_rate = 0.1, weights = None, metric = None, reg = None, l1_coef = 0.0, l2_coef = 0.0, sgd_sample = None, random_state = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = self.MAE if metric == 'mae' else self.MSE if metric == 'mse' else self.RMSE if metric == 'rmse' else self.MAPE if metric == 'mape' else self.R2 if metric == 'r2' else metric
        self.best_score = 0.0
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        ans = 'MyLineReg class: '
        for key in self.__dict__:
            ans += key + '=' + str(self.__dict__[key]) + ', '
        return ans[:-2]
    
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose = False) -> None:
        X = X.copy()
        X.insert(0, 'intercept', 1)
        self.weights = pd.Series(1, index = X.columns)
        random.seed(self.random_state)
        for iter in range(1, self.n_iter + 1):
            y_pred = X.dot(self.weights) #calculate predictions on whole dataset
            y_delta = y_pred - y 
            if self.sgd_sample: #stochastic gradient descent
                sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample if isinstance(self.sgd_sample, int) else int(self.sgd_sample * X.shape[0]))
                X_sample = X.iloc[sample_rows_idx]
                y_sample = y.iloc[sample_rows_idx]
                y_pred_sample = X_sample.dot(self.weights)
                y_delta_sample = y_pred_sample - y_sample 
                gradient = y_delta_sample.dot(X_sample) * 2 / len(y_delta_sample) #gradient of loss function
            else: #classic gradient descent
                gradient = y_delta.dot(X) * 2 / len(y_delta) #gradient of loss function
            loss = (y_delta ** 2).sum() / len(y_delta) #loss function
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
            if verbose and iter % verbose == 0:
                output = f'{iter} | loss: {loss}'
                if self.metric:
                    output += f' | {self.metric.__name__}: {self.metric(y, y_pred)}'
                if callable(self.learning_rate):
                    output += f' | learning rate: {self.learning_rate(iter)}'
                print(output)
        if self.metric:
            y_pred = X.dot(self.weights)
            self.best_score = self.metric(y, y_pred)

    def get_coef(self):
        return self.weights.to_numpy()[1:]
    
    def get_intercept(self):
        return self.weights.to_numpy()[0]
    
    def predict(self, X: pd.DataFrame):
        X = X.copy()
        X.insert(0, 'intercept', 1)
        return X.dot(self.weights)
    
    def get_best_score(self):
        return self.best_score
    
    def MAE(self, y: pd.Series, y_pred: pd.Series):
        return (y - y_pred).abs().sum() / len(y)
    
    def MSE(self, y: pd.Series, y_pred: pd.Series):
        return ((y - y_pred) ** 2).sum() / len(y)
    
    def RMSE(self, y: pd.Series, y_pred: pd.Series):
        return (((y - y_pred) ** 2).sum() / len(y)) ** (1 / 2)
    
    def MAPE(self, y: pd.Series, y_pred: pd.Series):
        return ((y - y_pred) / y).abs().sum() * 100 / len(y)
    
    def R2(self, y: pd.Series, y_pred: pd.Series):
        return 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


    
a = MyLineReg(learning_rate = lambda iter: 0.5 * (0.85 ** iter), metric = 'rmse', reg = 'elasticnet', l1_coef = 0.5, l2_coef = 0.5, sgd_sample = 0.1)
print(a)
#print(X, y)
a.fit(X, y, verbose = 10)
print(a.get_coef(), a.get_intercept())
print(a.get_best_score())
#print(a.predict(X))
#print(y)
    
