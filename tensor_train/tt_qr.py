import numpy as np

from core import TensorTrain
from numpy import reshape, dot, tensordot
from numpy.linalg.linalg import qr

def tt_qr(tt, op):
    rm = 1
    d = tt.d
    n = tt.n
    r = tt.r
    cores = [core[:] for core in tt.cores]
    if op == 'rl' or op == 'RL':
        # Perform right-to-left ortogonalization
        for k in xrange(d - 1, 0, -1):
            dims = cores[k].shape
            next_core = reshape(cores[k], (r[k], n[k] * r[k + 1]))
            cores[k], next_core = qr(next_core.T)
            cores[k] = cores[k].T
            next_core = next_core.T
            # Move dimension from right indices to left indices
            cores[k] = reshape(cores[k], (r[k], n[k], r[k + 1]))
            cores[k - 1] = tensordot(cores[k - 1], next_core, [len(cores[k - 1].shape) - 1, 0])
        if r[0] != 1:
            dims = cores[0].shape
            first, second = dims[0], dims[1] * dims[2]
            cores[0], rm = qr(reshape(cores[0], (first, second)).T)
            cores[0] = cores[0].T
            rm = rm.T
    elif op == 'lr' or op == 'LR':
        # Perform left-to-right ortogonalization
        for k in xrange(0, d - 1):
            dims = cores[k].shape
            first, second = r[k] * n[k], r[k + 1]
            next_core = reshape(cores[k], (first, second))
            cores[k], next_core = qr(next_core)
            cores[k] = reshape(cores[k], (r[k], n[k], r[k + 1]))
            cores[k + 1] = reshape(cores[k + 1], (r[k + 1], n[k + 1] * r[k + 2]))
            cores[k + 1] = np.dot(next_core, cores[k + 1])
            cores[k + 1] = reshape(cores[k + 1], (r[k + 1], n[k + 1], r[k + 2]))
        if r[-1] != 1:
            first, second = r[-2] * n[-1], r[-1]
            cores[-1], rm = qr(reshape(cores[-1], (first, second)))
    tt = TensorTrain()
    tt.d = d
    tt.r = r
    tt.n = n
    tt.cores = cores
    return tt