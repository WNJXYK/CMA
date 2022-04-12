from tqdm import trange
from copy import deepcopy
import scipy.linalg as linalg
import numpy as np
from sklm import sklm
from fastGFK import fastGFK

__all__ = ["CMA"]

class CMA:
    def __init__(self, model, **kwargs):
        self.model = model
        self.alpha = kwargs['alpha'] if 'alpha' in kwargs else None
        self.dim = kwargs['dim'] if 'dim' in kwargs else None
        self.mode = kwargs['mode'] if 'mode' in kwargs else 'csa'
        if self.mode not in ['cgfk', 'csa']: raise Exception("Mode must be cgfk / csa")
        
        if "debug" in kwargs:
            print("CMA: ")
            print(" > Mode: ", self.mode)
            print(" > Alpha: ", self.alpha)
            print(" > Dim: ", self.dim)

    def null(self, a, rtol=1e-5):
        u, s, v = np.linalg.svd(a)
        rank = (s > rtol*s[0]).sum()
        return v[rank:].T.copy()
    
    def init_reps(self, Xs):
        _, s, Vs = linalg.svd(Xs)
        if self.dim is None: self.dim = np.sum(s > 10)
        Ps = Vs.T[:, :self.dim]
        m = min(len(s), self.dim)
        self.Q = np.hstack([Ps, self.null(Ps.T)])
        self.Ut = Ps
        self.S = s[:m]
        self.mu = np.mean(Xs, axis=0, keepdims=True).T
        self.nprev = Xs.shape[0]
    
    def envo_reps(self, Xt):
        self.Ut, self.S, mu, self.nprev = sklm(Xt.T, self.Ut, self.S, self.mu, self.nprev, self.alpha, self.dim)
        if self.mode == 'csa': return np.dot(Xt, np.dot(self.Ut, self.Ut.T))
        G = fastGFK(self.Q, self.Ut)
        return np.dot(Xt, G)

    def predict(self, X):
        X = self.envo_reps(X)
        preds = self.model.predict(X)
        return preds
    
    def fit(self, Xs, ys):
        self.model.fit(Xs, ys.ravel())
        self.init_reps(Xs)
        return self.model
                


