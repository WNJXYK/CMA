import numpy as np
import scipy.linalg as linalg
__all__ = ["sklm"]

def init(data, mu0, K):
    '''
    This function is not tested, because CMA init its own basis, mean, singular values
    '''
    (m, n) = data.shape

    if n == 1: raise Exception("Not implemented when init with one sample")
    
    mu = np.mean(data, axis=1, keepdims=True)
    data = data - np.repeat(mu, n, axis=1)
    U, d, V = linalg.svd(data, full_matrices=m<=n)

    if not K is None:
        siz = min(K, len(d))
        d, U = d[: siz], U[:, :siz]

    return U, d, mu

def envolves(data, U0=None, D0=None, mu0=None, n0=None, ff=None, K=None):
    (m, n) = data.shape
    if not mu0 is None:
        mu1 = np.mean(data, axis=1, keepdims=True)
        data = data - np.repeat(mu1, n, axis=1)
        data = np.hstack([data, np.sqrt(n0 / (n + n0) * n) * (mu0 - mu1)])
        mu = ff * n0 / (n + ff * n0) * mu0  + n / (n + ff * n0) * mu1 # This line is modified due to overflow problem
        n = n + ff * n0
    
    data_proj = np.dot(U0.T, data)
    data_res  = data - np.dot(U0, data_proj)
    q, _ = linalg.qr(data_res, mode="economic")
    
    Q = np.hstack([U0, q])
    R = np.vstack([
        np.hstack([ ff * np.diag(D0), data_proj ]), 
        np.hstack([ np.zeros((data.shape[1], D0.shape[0])), np.dot(q.T, data_res) ])
    ])

    U, d, V = linalg.svd(R, full_matrices=R.shape[0]<=R.shape[1])
    
    if K is None:
        cutoff = np.sum(d ** 2) * 1e-6
        siz = 0
        for i in range(len(d)):
            siz = i
            if d[i] ** 2 < cutoff: break
    else:
        siz = min(K, len(d), n)
    d = d[: siz]
    U = np.dot(Q, U[:, :siz])

    return U, d, mu, n

def sklm(data, U0=None, D0=None, mu0=None, n0=None, ff=None, K=None):
    data = np.array(data)
    (m, n) = data.shape

    if U0 is None or D0 is None:
        U, d, mu = init(data, mu0, K)
    else:
        if ff is None: ff = 1.0
        if n0 is None: n0 = n
        U, d, mu, n = envolves(data, U0, D0, mu0, n0, ff, K)
    
    return U, d, mu, n
        

