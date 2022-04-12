import numpy as np
import scipy.linalg as linalg
import scipy.sparse.linalg as slinalg

__all__ = ["faskGFK"]

def fastGFK(Q, Pt):
    N = Q.shape[1]
    dim = Pt.shape[1]

    Ps = Q[:, :dim]
    R  = Q[:, dim:]
    A = np.dot(Ps.T, Pt)
    B = np.dot(R.T, Pt)

    Binv = linalg.pinv(B)
    if dim >= min(np.dot(A, Binv).shape): 
        V1, s, V2 = linalg.svd(np.dot(A, Binv))
    else:
        V1, s, V2 = slinalg.svds(np.dot(A, Binv), dim)
        if len(s) < dim: V1, s, V2 = linalg.svd(np.dot(A, Binv))
    V2 = -V2.T

    theta = np.arctan(1.0 / s).real
    
    eps = 1e-20
    divs = 2.0 * max(max(theta), eps)
    B1 = 0.5 * np.diag(1 + np.sin(2 * theta) / divs)
    B2 = 0.5 * np.diag( (-1 + np.cos(2 * theta)) / divs)
    B3 = B2
    B4 = 0.5 * np.diag(1 - np.sin(2 * theta) / divs)

    PsV1 = np.dot(Q[:, :dim], V1)
    RsV2 = np.dot(Q[:, dim:], V2[:, :dim])
    
    Qv = np.hstack([PsV1, RsV2])
    Bsub = np.vstack([
        np.hstack([B1, B2]),
        np.hstack([B3, B4]),
    ])
    Q1 = Qv[:2*dim, :2*dim]
    Q3 = Qv[2*dim:, :2*dim]

    Q3_Bsub = np.dot(Q3, Bsub)
    Q1_Bsub_Q1t = np.dot(np.dot(Q1, Bsub), Q1.T)
    Q3_Bsub_Q1t = np.dot(Q3_Bsub, Q1.T)
    Q3_Bsub_Q3t = np.dot(Q3_Bsub, Q3.T)

    G = np.vstack([
        np.hstack([Q1_Bsub_Q1t, Q3_Bsub_Q1t.T]),
        np.hstack([Q3_Bsub_Q1t, Q3_Bsub_Q3t]),
    ])

    return G
