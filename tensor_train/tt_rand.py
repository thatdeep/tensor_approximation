import numpy as np
from core import TensorTrain

def tt_rand(n, d, r):
    if not hasattr(n, "__len__"):
        n = np.ones(d) * n
    if not hasattr(r, "__len__"):
        r = np.ones(d+1) * r
        r[0] = 1
        r[d] = 1
    tt = TensorTrain()
    tt.n = n
    tt.r = r
    tt.d = d
    tt.cores = []
    for left, mid, right in zip(r[:d], n, r[1:]):
        tt.cores.append(np.random.random((left, mid, right)))
    return tt