import numpy as np
import math

from core import TensorTrain
from utils import frobenius_norm, rank_chop
from numpy import reshape, dot, tensordot
from numpy.linalg.linalg import svd, qr


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
        r_new = B.r[k+1]
        B.cores[k] = B.cores[k][:, :r_new]
        B.cores[k] = reshape(B.cores[k], (B.r[k], B.n[k], B.r[k+1]))
        V = V[:r_new]
        print '>>>', s.size, r_new
        s = s[:r_new]
        B.cores[k+1] = tensordot(dot(np.diag(s), V), B.cores[k+1], [1, 0])
    return B