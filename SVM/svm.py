import numpy as np
import copy
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform

class SVM:
    def __init__(self, variant='primal', C=1, a=0.01, r=0.01, epoch=10, tol=1e-6, default_schedule=True, kernel='linear'):
        self.variant = variant
        self.C = C
        self.a = a
        self.r = r
        self.epoch = epoch
        self.tol = tol
        self.w = None
        self.b = None
        self.y = None
        self.X = None
        self.alpha = None  
        self.default_schedule = default_schedule
        self.kernel = kernel
        self.support_vectors = None
        self.support_labels = None

    def fit(self, X, y):
        _X = copy.deepcopy(X)
        _y = copy.deepcopy(y)
        if self.variant == 'primal':
            self._fit_primal(_X, _y)
        elif self.variant == 'dual':
            self._fit_dual(_X, _y)
        else:
            raise ValueError('Invalid variant')
        
    def _fit_primal(self, X, y):
        self.w = np.zeros(X.shape[1])
        w_0 = np.zeros((X.shape[1] - 1))
        N = len(y)
        i = 0
        prev_w = None
        converged = False
        while i < self.epoch and not converged:
            perm = np.random.permutation(len(y))
            X = X[perm]
            y = y[perm]
            for j in range(len(X)):
               y_predicted = np.dot(X[j], self.w.T)
               prev_w = copy.deepcopy(self.w)

               if (y_predicted * y[j]) <= 1:
                # subgradient = np.insert(w_0, 0, 0) - self.C * N * y[j] * X[j]
                # self.w = prev_w - self.r * subgradient
                self.w = prev_w - self.r * np.insert(w_0, 0, 0) + self.r * self.C * N * y[j] * X[j]
               else:
                    w_0 = (1 - self.r)*w_0

            if self.default_schedule:
                self.r = (self.r)/(1 + ((self.r/self.a)*(i+1)))
            else:
                self.r = (self.r)/(1 + (i+1))
            i += 1
            converged = np.linalg.norm(self.w - prev_w) < self.tol


    # def gaussian_kernel(self, X1, X2):
    #     distance_matrix = euclidean_distances(X1, X2)
    #     return np.exp(-(distance_matrix**2) / (self.r)) 

    def gaussian_kernel(self, x1, x2):
        return np.exp(-(np.linalg.norm(x1 - x2) ** 2 / (self.r)))
    
    def gaussian_kernel_matrix(self, X):
        pairwise_distances = squareform(pdist(X, 'euclidean'))
        K = np.exp(-pairwise_distances / (self.r))
        return K
    
    def _get_bias(self, alpha, X, y):
        # support vectors (alphas > 0)
        support_vectors_indices = np.where(alpha > 0)[0]
        j = support_vectors_indices[0]
        # b = np.mean([y[j] - np.sum(alphas[support_vectors_indices] * y[support_vectors_indices] * np.array([kernel(X[i], X[j]) for i in support_vectors_indices])) for j in support_vectors_indices])

        b = y[j] - np.sum(alpha[support_vectors_indices] * y[support_vectors_indices] * np.array([self.gaussian_kernel(X[i], X[j]) for i in support_vectors_indices]))
        return b

    def _dual_obj_fun(self, X, y):
        K = None
        if self.kernel == 'linear':
            K = np.dot(X, X.T)
        elif self.kernel == 'gaussian':
            distance_matrix = euclidean_distances(X, X)
            K = np.exp(-(distance_matrix**2) / (self.r))
        else:
            raise ValueError("Kernel not recognized")
        return lambda alpha: 0.5 * np.sum((alpha * y * alpha * y * K) - np.sum(alpha))
        
        
        # return lambda alpha: (0.5 * np.sum(np.outer(alpha * y, alpha * y) * K) - np.sum(alpha))
    
        
    def _fit_dual(self, X, y):
        # dual_obj_fun = lambda alpha: (0.5 * np.sum(alpha * alpha * y * y * np.array([self._kernel(X[i], X[j]) for i in range(len(X)) for j in range(len(X))]).reshape(len(X), len(X))) - np.sum(alpha))
        
        # fun = self._dual_obj_fun(X, y)

        # K = np.dot(X, y.T)  # Kernel matrix
        K = None
        if self.kernel == 'linear':
            K = np.dot(X, X.T)
        elif self.kernel == 'gaussian':
            pairwise_distances = squareform(pdist(X, 'euclidean'))
            K = np.exp(-pairwise_distances / (self.r))

        dual_objective = lambda alpha: - (np.sum(alpha) - 0.5 *  np.sum(np.outer(alpha * y, alpha * y) * K))

        
        constraints = ({'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)})
        bounds = [(0, self.C) for _ in range(len(X))]
        alpha = np.zeros(len(X))

        result = minimize(dual_objective, alpha, bounds=bounds, constraints=constraints)
        optimal_alpha = result.x

        if self.kernel == 'linear':
            self.w = np.sum([optimal_alpha[i] * y[i] * X[i] for i in range(len(X))], axis=0)
            self.b = np.sum([y[i] - np.dot(self.w.T, X[i]) for i in range(len(X))]) / len(X)
            # self.w = np.insert(self.w, 0, self.b)
            self.alpha = optimal_alpha
            self.support_vectors = [X[i] for i in range(len(X)) if self.alpha[i] > 0]
            self.support_labels = [y[i] for i in range(len(X)) if self.alpha[i] > 0]
            
        elif self.kernel == 'gaussian':
            self.alpha = optimal_alpha
            self.b = self._get_bias(self.alpha, X, y)
            sv_indices = np.where(optimal_alpha > 0)[0]
            self.support_vectors = X[sv_indices]
            self.support_labels = y[sv_indices]
            print("Number of support vectors: ", len(self.support_vectors))
            print("Support vectors: ", self.support_vectors)
            self.y = y
            self.X = X

    def get_support_vectors(self):
        return self.support_vectors
    
    def get_model(self):
        if self.variant == 'primal':
            return self.w
        elif self.variant == 'dual' and self.kernel == 'linear':
            weights = copy.deepcopy(self.w)
            weights = np.insert(weights, 0, self.b)
            return weights

    def predict(self, X):
        if self.variant == 'primal':
            return self._primal_predict(X)
        elif self.variant == 'dual':
            return self._dual_predict(X)
        
    def _primal_predict(self, X):
        return np.sign(np.dot(X, self.w.T))
    
    def _dual_predict(self, X):
        if self.kernel == 'linear':
            return np.sign(np.dot(X, self.w.T) + self.b)
        elif self.kernel == 'gaussian':
            predictions = []
            for x in X:
                prediction = 0
                for i in range(len(self.support_vectors)):
                    prediction += (self.alpha[i] * self.support_labels[i] * self.gaussian_kernel(self.support_vectors[i], x)) + self.b
                predictions.append(prediction)
            return np.sign(predictions)
        else:
            raise ValueError("Kernel not recognized")
    