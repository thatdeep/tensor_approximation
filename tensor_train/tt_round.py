import numpy as np
import math

from core import TensorTrain
from utils import rank_chop
from numpy import reshape, dot, tensordot
from numpy.linalg.linalg import svd, qr
from tt_basic_algebra import tt_zeros
from utils import frobenius_norm


def tt_round(tt, eps=1e-10):
    tt_norm = frobenius_norm(tt)
    d = tt.d
    n = tt.n
    delta = eps / math.sqrt(d - 1) * tt_norm

    B = TensorTrain(tt)
    B.r = np.zeros_like(tt.r)

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
    diag_sets = [0]*d

    dist_norm = tt_norm

    # Suppress
    for k in xrange(d-1):
        dims = B.cores[k].shape
        first, second = reduce(lambda x, y: x*y, dims[:-1]), dims[-1]
        cur_core = reshape(B.cores[k], (first, second))
        B.cores[k], s, V = svd(cur_core, full_matrices=False)
        B.r[k+1] = rank_chop(s, delta)
        r_new = B.r[k + 1]
        diag_sets[k] = r_new
        if r_new == 0:
            # Tensor becomes zero as convolution by rank 0
            return tt_zeros(n)
        B.cores[k] = B.cores[k][:, :r_new]
        B.cores[k] = reshape(B.cores[k], (B.r[k], B.n[k], B.r[k+1]))
        V = V[:r_new]
        #print '>>>', s.size, r_new
        s = s[:r_new]
        #s_back, s_forward = s**(1. / (d - k)), s**(1.*(d - k - 1)/(d-k))
        #if k == d - 2:
        #    tt_norm = np.linalg.norm(s)
        #B.cores[k] = np.dot(B.cores[k], np.diag(s_back))
        B.cores[k] = reshape(B.cores[k], (B.r[k], B.n[k], B.r[k+1]))
        B.cores[k+1] = tensordot(dot(np.diag(s), V), B.cores[k+1], [1, 0])
        dist_norm = np.linalg.norm(s)
    # Now cores[-1] contains all frobenius norm of a whole tensor
    #distributed_norm = tt_norm / d
    # Firstly, make last core norm lower
    #B.cores[-1][:r_new, :] /= (tt_norm / distributed_norm)
    # And after that, distribute it's norm around all other cores
    #for i in xrange(d-1):
    #    B.cores[i][:, :diag_sets[i]] *= distributed_norm
    #print [frobenius_norm(core) for core in B.cores]

    return B

def tt_round_modified(tt, eps=1e-10):
    tt_norm = frobenius_norm(tt)
    d = tt.d
    n = tt.n
    delta = eps / math.sqrt(d - 1) * tt_norm

    B = TensorTrain(tt)
    B.r = np.zeros_like(tt.r)

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
    diag_sets = [0]*d

    dist_norm = tt_norm

    # Suppress
    for k in xrange(d-1):
        dims = B.cores[k].shape
        first, second = reduce(lambda x, y: x*y, dims[:-1]), dims[-1]
        cur_core = reshape(B.cores[k], (first, second))
        B.cores[k], s, V = svd(cur_core, full_matrices=False)
        B.r[k+1] = rank_chop(s, eps / math.sqrt(d - 1) * dist_norm)
        r_new = B.r[k + 1]
        diag_sets[k] = r_new
        if r_new == 0:
            # Tensor becomes zero as convolution by rank 0
            return tt_zeros(n)
        B.cores[k] = B.cores[k][:, :r_new]
        B.cores[k] = reshape(B.cores[k], (B.r[k], B.n[k], B.r[k+1]))
        V = V[:r_new]
        #print '>>>', s.size, r_new
        s = s[:r_new]
        s_back, s_forward = s**(1. / (d - k)), s**(1.*(d - k - 1)/(d-k))
        #if k == d - 2:
        #    tt_norm = np.linalg.norm(s)
        B.cores[k] = np.dot(B.cores[k], np.diag(s_back))
        B.cores[k] = reshape(B.cores[k], (B.r[k], B.n[k], B.r[k+1]))
        B.cores[k+1] = tensordot(dot(np.diag(s_forward), V), B.cores[k+1], [1, 0])
        dist_norm = np.linalg.norm(s_forward)
    # Now cores[-1] contains all frobenius norm of a whole tensor
    #distributed_norm = tt_norm / d
    # Firstly, make last core norm lower
    #B.cores[-1][:r_new, :] /= (tt_norm / distributed_norm)
    # And after that, distribute it's norm around all other cores
    #for i in xrange(d-1):
    #    B.cores[i][:, :diag_sets[i]] *= distributed_norm
    #print [frobenius_norm(core) for core in B.cores]

    return B