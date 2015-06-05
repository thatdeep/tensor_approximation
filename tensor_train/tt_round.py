import numpy as np
import math

from core import TensorTrain
from utils import frobenius_norm, rank_chop
from numpy import reshape, dot, tensordot
from numpy.linalg.linalg import svd, qr

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
            first, second = dims[0], dims[1] * dims[2]
            K = min(first, second)
            next_core = reshape(cores[k], (first, second))
            cores[k], next_core = qr(next_core.T)
            cores[k] = cores[k].T
            next_core = next_core.T
            # Move dimension from right indices to left indices
            cores[k] = reshape(cores[k], (cores[k].shape[0], n[k], cores[k].shape[1] / n[k]))
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


def tt_round(tt, eps=1e-9):
    d = tt.d
    n = tt.n
    delta = eps / math.sqrt(d - 1)

    B = TensorTrain()
    B.d = tt.d
    B.n = tt.n
    B.r = np.zeros_like(tt.r)
    B.cores = [core[:] for core in tt.cores]

    for k in xrange(d - 1, 0, -1):
        # Find reshaping dimensions
        dims = B.cores[k].shape
        first, second = dims[0], reduce(lambda x, y: x*y, dims[1:])
        K = min(first, second)
        # Get qr of transposed core
        cur_core = reshape(B.cores[k], (first, second))
        B.cores[k], cur_core = qr(cur_core.T)
        B.cores[k] = B.cores[k].T
        cur_core = cur_core.T

        # move dimension from right indices to left indices
        B.cores[k] = reshape(B.cores[k], (B.cores[k].shape[0], B.n[k], B.cores[k].shape[1] / B.n[k]))

        # Do we really shall reshape or simply stay Q^t as a matrix?
        #B.cores[k] = reshape(B.cores[k], (K,) + second)
        B.cores[k-1] = tensordot(B.cores[k-1], cur_core, [len(B.cores[k-1].shape) - 1, 0])

    B.r[0] = 1
    B.r[-1] = 1

    # Suppress
    for k in xrange(d-1):
        dims = B.cores[k].shape
        first, second = reduce(lambda x, y: x*y, dims[:-1]), dims[-1]
        cur_core = reshape(B.cores[k], (first, second))
        B.cores[k], s, V = svd(cur_core, full_matrices=False)
        B.r[k+1] = rank_chop(s, delta)
        r_new = B.r[k + 1]
        B.cores[k] = B.cores[k][:, :r_new]
        B.cores[k] = reshape(B.cores[k], (B.r[k], B.n[k], B.r[k+1]))
        V = V[:r_new]
        print '>>>', s.size, r_new
        s = s[:r_new]
        B.cores[k+1] = tensordot(dot(np.diag(s), V), B.cores[k+1], [1, 0])
    return B