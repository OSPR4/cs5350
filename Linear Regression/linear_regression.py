import numpy as np
import copy
from scipy import linalg as LA

class LinearRegression:
    def __init__(self, grad_variant='batch', r=0.01, max_iter=float("inf"), tol=1e-6):
        self.grad_variant = grad_variant
        self.r = r
        self.max_iter = max_iter
        self.tol = tol
        # self.X = X
        # self.y = y
        self.w = None

    def get_weights(self):
        return self.w

    def fit(self, X, y):
        self._regression(X, y)
        
        
    def _regression(self, X, y):
        converged = False
        w = np.zeros(X.shape[1])
        interation = 0

        while not converged or interation < self.max_iter:
            # print("\n" + "w before calculate_gradient" + str(W))
            gradient_J_W_t = np.zeros(X.shape[1])
            if self.grad_variant == 'batch':
                gradient_J_W_t = self.calculate_gradient_batch(X, y, w)
            elif self.grad_variant == 'stochastic':
                gradient_J_W_t = self.calculate_gradient_stochastic(X, y, w)
            else:
                raise ValueError('Invalid variant of gradient descent')
            
            w_t = w - (self.r * gradient_J_W_t)
            converged = np.linalg.norm(w_t - w) < self.tol
            w = w_t
            interation += 1
        self.w = w
    
    def calculate_gradient_batch(self, X, y, w):
        gradient = np.zeros_like(w)
        W_transpose = np.transpose(w)
        for j in range(len(w)):
            for i in range(len(X)):
                gradient[j] += (y[i] - np.dot(W_transpose, X[i])) * (X[i][j] * -1)
        return gradient
    
    def calculate_gradient_stochastic(self, X, y, w):
        gradient = np.zeros_like(w)
        random_x = np.random.randint(0, len(X))
        sample_x = X[random_x]
        sample_y = y[random_x]
        for j in range(len(w)):
            gradient[j] += (sample_y - np.dot(sample_x, w.T)) * (sample_x[j] * -1)

        return gradient

    def predict(self, X):
        return X.dot(self.w)