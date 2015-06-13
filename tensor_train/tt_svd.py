import numpy as np
import math

from utils import frobenius_norm, rank_chop
from numpy import reshape, dot
from numpy.linalg.linalg import svd
from tt_basic_algebra import tt_zeros


# TODO make norm redistribution
def tt_svd(tt, A, eps=1e-9):
    d = len(A.shape)
    tt.n = A.shape
    frob_norm = frobenius_norm(A)
    delta = frob_norm * eps / math.sqrt(d - 1)
    N = A.size
    ns = np.array(A.shape)
    C = A
    tt.cores = []
    ranks = np.zeros(d + 1, dtype=np.int)
    ranks[0] = 1
    for k in xrange(d - 1):
        C = reshape(C, (ranks[k] * ns[k], N / (ranks[k] * ns[k])))
        U, s, V = svd(C, full_matrices=False)
        ranks[k + 1] = rank_chop(s, delta)
        r_new = ranks[k + 1]
        if r_new == 0:
            # Tensor becomes zero as convolution by rank 0
            tt = tt_zeros(tt.n)
        U = U[:, :ranks[k + 1]]
        tt.cores.append(reshape(U, (ranks[k], ns[k], r_new)))
        V = V[:r_new, :]
        s = s[:r_new]
        V = dot(np.diag(s), V)
        C = V
        N = N * r_new / (ns[k] * ranks[k])
        r = r_new
    tt.cores.append(C.reshape(C.shape + (1,)))
    tt.d = d
    tt.r = ranks
    tt.r[-1] = 1