import math
import numpy as np
import mpmath as mp

from mpmath import mpf, workdps
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
    dps = 256
    with workdps(dps):
        nrm = mpf('1.0')
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
        c = frobenius_norm(cur_core)
        cur_core /= c
        with workdps(dps):
            nrm *= c

        # move dimension from right indices to left indices
        cores[k] = reshape(cores[k], (r[k], n[k], r[k+1]))

        # Convolve cores[k-1] with R^t
        cores[k-1] = reshape(cores[k-1], (r[k-1] * n[k-1], r[k]))
        cores[k-1] = dot(cores[k-1], cur_core)
        cores[k-1] = reshape(cores[k-1], (r[k-1], n[k-1], r[k]))
    c = frobenius_norm(cores[0])
    cores[0] /= c
    with workdps(dps):
        nrm *= c
    # New compression ranks
    sr = np.zeros_like(r)
    sr[0] = sr[-1] = 1

    delta = eps / math.sqrt(d - 1)
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
    t = TensorTrain.from_cores(cores, reuse=True, norm=nrm)
    return t

'''
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
    dps = 256
    with workdps(dps):
        nrm = mpf('1.0')
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
        c = frobenius_norm(cur_core)
        cur_core /= c
        with workdps(dps):
            nrm *= c

        # move dimension from right indices to left indices
        cores[k] = reshape(cores[k], (r[k], n[k], r[k+1]))

        # Convolve cores[k-1] with R^t
        cores[k-1] = reshape(cores[k-1], (r[k-1] * n[k-1], r[k]))
        cores[k-1] = dot(cores[k-1], cur_core)
        cores[k-1] = reshape(cores[k-1], (r[k-1], n[k-1], r[k]))
    c = frobenius_norm(cores[0])
    cores[0] /= c
    with workdps(dps):
        nrm *= c
    # New compression ranks
    sr = np.zeros_like(r)
    sr[0] = sr[-1] = 1

    delta = eps / math.sqrt(d - 1)
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
    t = TensorTrain.from_cores(cores, reuse=True)
    t.norm = nrm
    t._normed = True
    return t
'''