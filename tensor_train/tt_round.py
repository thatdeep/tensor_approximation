import math
import numpy as np

from core import TensorTrain
from numpy import reshape, dot
from tt_basic_algebra import tt_zeros
from utils import frobenius_norm, csvd

from numpy.linalg import qr


def tt_round(t, eps=1e-10, reuse=False):
    """
    Lowers compression ranks of tensor t in TT-format. Resulting tensor B has property
    that || B - t || < eps || t ||. Rewrites tensor t if reuse parameter set to True.

    Parameters
    ----------
    t : TensorTrain
        tensor in TT-format
    eps : float, optional
        error bound of rank compression, defaults to 1e-10
    reuse : bool, optional
        reuse data flag. Rewrite cores of t, if set to True. By default set to False.

    Returns
    -------
    b : TensorTrain
        tensor in TT-format with lower compression ranks
    """
    norm = frobenius_norm(t)
    d = t.d
    n = t.n
    r = t.r
    delta = eps / math.sqrt(d - 1) * norm

    # new cores for rounded tensor
    cores = [core for core in t.cores] if reuse else [core.copy() for core in t.cores]

    # Ortogonalization process
    for k in xrange(d - 1, 0, -1):
        # Find reshaping dimensions
        cur_core = reshape(cores[k], (r[k], n[k] * r[k+1]))

        # Get qr of transposed core
        cores[k], cur_core = qr(cur_core.T)
        cores[k] = cores[k].T
        cur_core = cur_core.T

        # move dimension from right indices to left indices
        cores[k] = reshape(cores[k], (r[k], n[k], r[k+1]))

        # Convolve cores[k-1] with R^t
        cores[k-1] = reshape(cores[k-1], (r[k-1] * n[k-1], r[k]))
        cores[k-1] = dot(cores[k-1], cur_core)
        cores[k-1] = reshape(cores[k-1], (r[k-1], n[k-1], r[k]))

    # New compression ranks
    sr = np.zeros_like(r)
    sr[0] = sr[-1] = 1

    # Compression process
    for k in xrange(d-1):
        cur_core = reshape(cores[k], (sr[k] * n[k], r[k+1]))
        # Perform svd with rank chop
        cores[k], s, V = csvd(cur_core, delta)
        if not s.size:
            # Oops, we have a zero tensor as a convolution by rank 0
            return tt_zeros(n)
        sr[k+1] = s.size
        cores[k] = reshape(cores[k], (sr[k], n[k], sr[k+1]))

        cores[k+1] = reshape(cores[k+1], (r[k+1], n[k+1] * r[k+2]))
        cores[k+1] = dot(dot(np.diag(s), V), cores[k+1])
        cores[k+1] = reshape(cores[k+1], (sr[k+1], n[k+1], r[k+2]))
    return TensorTrain.from_cores(cores, reuse=True)

"""
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
"""