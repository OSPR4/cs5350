import numpy as np
import copy

class Perceptron:
    def __init__(self, variant='standard', epoch=10, r=0.01) -> None:
        self.variant = variant
        self.epoch = epoch
        self.w = None
        self.r = r
        self.w_list = []
        self.c_list = []

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        _X = copy.deepcopy(X)
        _y = copy.deepcopy(y)
        if self.variant == 'standard':
            self._fit_standard(_X, _y)
        elif self.variant == 'voted':
            self._fit_voted(_X, _y)
        elif self.variant == 'average':
            self._fit_average(_X, _y)
        else:
            raise ValueError('Invalid variant')
        
    def predict(self, X):
        if self.variant == 'standard':
            return self._predict_standard(X)
        elif self.variant == 'voted':
            return self._predict_voted(X)
        elif self.variant == 'average':
            return self._predict_average(X)

        
    def _fit_standard(self, X, y):
        for i in range(self.epoch):
            perm = np.random.permutation(len(y))
            X = X[perm]
            y = y[perm]
            for j in range(len(X)):
                y_predicted = np.sign(np.dot(X[j], self.w.T))
                if y_predicted != y[j]:
                    self.w = self.w + self.r * (y[j] * X[j])
    
    def _predict_standard(self, X):
        return np.sign(np.dot(X, self.w.T))
    
    def _fit_voted(self, X, y):
        w = np.zeros(X.shape[1])
        m = 0
        c = 1
        for i in range(self.epoch):
            for j in range(0, len(X)):
                y_predicted = np.sign(np.dot(X[j], w.T))
                if y_predicted != y[j]:
                    self.w_list.insert(m, copy.deepcopy(w))
                    self.c_list.insert(m, c)
                    w = w + self.r * (y[j] * X[j])
                    m += 1
                    c = 1
                else:
                    c += 1
        self.w_list.insert(m, copy.deepcopy(w))
        self.c_list.insert(m, c)

    def _predict_voted(self, X):
        y_predicted = np.zeros(len(X))
        for i in range(len(X)):
            for j in range(len(self.w_list)):
                y_predicted[i] += self.c_list[j] * np.sign(np.dot(X[i], self.w_list[j].T))
        return np.sign(y_predicted) 
    
    def _fit_average(self, X, y):
        w = np.zeros(X.shape[1])
        a = np.zeros(X.shape[1])

        for i in range(self.epoch):
            for j in range(len(X)):
                y_predicted = np.sign(np.dot(X[j], w.T))
                if y_predicted != y[j]:
                    w = w + self.r * (y[j] * X[j])
                a += w
        self.w = copy.deepcopy(a)

    def _predict_average(self, X):
        return np.sign(np.dot(X, self.w.T))