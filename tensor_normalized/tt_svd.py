import math
import numpy as np

from numpy import reshape, dot
from mpmath import mpf, workdps
from constants import TENSOR_NORM_DPS
from tt_basic_algebra import tt_zeros
from utils import frobenius_norm, csvd


def tt_svd(t, a, eps=1e-9):
    d, shape, size = len(a.shape), a.shape, a.size
    # think about mpf
    norm = frobenius_norm(a)
    delta = eps / math.sqrt(d - 1) * norm
    C = a
    cores = []
    r = np.zeros(d+1, dtype=int)
    r[0] = r[-1] = 1

    for k in xrange(d-1):
        C = reshape(C, (r[k] * shape[k], size / (r[k] * shape[k])))
        # Perform svd with rank chop
        U, s, V = csvd(C, delta)
        if k == 0:
            s /= norm
        if not s.size:
            # Oops, we have a zero tensor as a convolution by rank 0
            return tt_zeros(shape)
        r[k+1] = s.size
        cores.append(reshape(U, (r[k], shape[k], r[k+1])))
        C = dot(np.diag(s), V)
        size = size * r[k+1] / (shape[k] * r[k])
    cores.append(C[..., np.newaxis])

    t.d = d
    t.r = r
    t.shape = shape
    with workdps(TENSOR_NORM_DPS):
        t.norm = mpf(norm)
    t.cores = cores