# CMA

A Python Implementation of 'Continuous Manifold-based Adaptation', the official MatLab version: [jhoffman/cma](https://github.com/jhoffman/cma).

Judy Hoffman, Trevor Darrell, Kate Saenko: Continuous Manifold Based Adaptation for Evolving Visual Domains. CVPR 2014: 867-874

# How to use

```py3
from sklearn.svm import LinearSVC
from cma import CMA

# Define a CMA module with Linear SVM
# Mode is 'cgfk' (cgfk / csa)
# Alpha is 1.5 - Forgetting parameter for online subspace learning
# Dim is 10
cma = CMA(LinearSVC(), **{'alpha': 1.5, 'dim': 10, 'mode': 'cgfk'})

# Init on source domain
cma.fit(Xs, ys.ravel())

# Envolves on data stream
for Xt in data_steam:
    yt = cma.predict(Xt)
```

We provide a [Notebook](https://github.com/WNJXYK/CMA/blob/main/CMA-Caltran.ipynb) to reproduce the default experiment in the official Matlab code.

# Experiments

Here is the experiment setting and hyper-parameters.
```
Dataset: caltran_gist
Norm_type: L1 Zscore
Size of Source Domain: 50
Size of Streaming: 480
Block Size: 2
Alpha: 1.5
Dim: 10
```

## Original Matlab Version

|StartIdx|  KNN |   SVM | KNN_cgfk | KNN_csa | SVM_cgfk | SVM_csa |
| ---: | ----: | -------: | ------: | -------: | ------: | -----:|
| 350  | 65.49 | 77.75 | 64.66    | 64.45   | 83.99    | 83.58   |
| 400  | 65.70 | 71.93 | 66.53    | 66.32   | 73.39    | 73.80   |
| 450  | 55.30 | 70.48 | 55.30    | 54.89   | 72.77    | 72.56   |
| 500  | 54.89 | 71.93 | 55.51    | 55.51   | 67.98    | 67.98   |
| 550  | 67.57 | 71.52 | 62.99    | 63.41   | 79.21    | 79.21   |
| Mean | 61.79 | 72.72 | 61.00    | 60.91   | 75.47    | 75.43   |

## This Python Implementation

|StartIdx|  KNN |   SVM | KNN_cgfk | KNN_csa | SVM_cgfk | SVM_csa |
| ---: | ----: | -------: | ------: | -------: | ------: | -----:|
|  350 | 63.96 |    77.50 |   66.46 |    69.17 |   84.79 | 84.79 |
|  400 | 65.21 |    72.08 |   64.17 |    64.17 |   73.96 | 74.17 |
|  450 | 56.46 |    69.58 |   56.67 |    56.88 |   72.50 | 72.71 |
|  500 | 56.04 |    71.88 |   52.92 |    53.54 |   66.25 | 67.92 |
|  550 | 55.00 |    71.67 |   55.00 |    53.96 |   76.25 | 79.38 |
| Mean | 59.33 |    72.54 |   59.04 |    59.54 |   74.75 | 75.79 |



