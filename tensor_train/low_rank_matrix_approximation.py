__author__ = 'Const'

import math
import numpy as np

from utils import rank_chop, frobenius_norm
from tensor_train import maxvol, TensorTrain, rmatdiv
from numpy import dot, reshape, tensordot

from numpy.linalg import qr, svd, inv


# This algo works if A fits in memory, and two matrices of size A fits in memory
def low_rank_matrix_approx(A, r, delta=1e-6):
    m, n = A.shape
    J = np.array(range(r))
    approx_prev = np.zeros_like(A)

    while True:
        R = A[:, J]
        Q, T = qr(R)
        assert Q.shape == (m, r)
        I = maxvol(Q)
        C = A[I, :].T
        assert C.shape == (n, r)
        Q, T = qr(C)
        assert Q.shape == (n, r)
        J = maxvol(Q)
        QQ = Q[J, :]
        # We need to store the same as A matrix
        approx_next = np.dot(A[:, J], np.dot(Q, np.linalg.inv(QQ)).T)
        if np.linalg.norm(approx_next - approx_prev) > delta * np.linalg.norm(approx_prev):
            return I, J
        approx_prev = approx_next


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


def low_rank(A, dims, r=None, eps=1e-9):
    d = len(dims)
    if r == None:
        r = np.ones(d + 1) * 2
    y = tt_rand(dims, d, r)
    y_rank = y.r
    (y, rm) = qr(y)
    y = rm * y

    swp = 1
    rmatrices = []
    rmatrices[0] = 1
    rmatrices[-1] = 1

    indices = [0]*d + 1
    indices[0] = np.zeros(0, y_rank[d])
    indices[-1] = np.zeros(y_rank[0], 0)
    r1 = 1

    for i in xrange(d - 1, 1, -1):
        core = y[i]
        core = reshape(core, (y_rank[i] * dims[i], y_rank[i+1]))
        core = np.dot(core, r1)
        core = reshape(core, (y_rank[i], dims[i] * y_rank[i+1])).T
        core, rm = qr(core)
        idx = maxvol(core)
        idx_prev = indices[i+1]
        rnew = min(dims[i] * y_rank[i+1], y_rank[i])
        idx_new = np.zeros((d-i+1, rnew))
        for s in xrange(rnew):
            f_idx = idx[s]
            # !!!
            rs, js = np.unravel_index([y_rank[i+1], dims[i]], f_idx)
            idx_new[:, s] = np.array([js] + idx_prev[:, rs])
        indices[i] = idx_new
        r1 = core[idx, :]
        core = rmatdiv(core, r1).T
        r1 = np.dot(r1, rm).T
        y[i] = reshape(core, (y_rank[i], dims[i], y_rank[i+1]))
        core = reshape(core, (y_rank[i] * dims[i], y_rank[i+1]))
        core = np.dot(core, rmatrices[i+1])
        core = reshape(core, (y_rank[i], dims[i] * y_rank[i+1])).T
        _, rm = qr(core)
        rmatrices[i] = rm
    core = y[0].reshape(core, (y_rank[0] * dims[0], y_rank[1]))
    return indices, rmatrices, y

"""
def subcore(F, dims, row_set, column_set, dir='make_rows', k=None):
    from itertools import product

    if dir == 'make_rows':
        row_direction = True
        assert len(column_set) == len(dims) - 1
        if k == None:
            k = len(row_set)
    elif dir == 'make_columns':
        row_direction = False
        assert len(row_set) == len(dims) - 1
        if k == None:
            k = len(column_set)
    else:
        raise Exception('wrong direction')

    if dir == 'make_rows':
        multi_index_set = row_set + [np.arange(dims[k])] + column_set[k:]
        left_dim = np.prod(row_set)
        right_dim = np.prod(column_set[k:])
    else:
        multi_index_set = row_set[:k] + [np.arange(dims[k])] + column_set
        left_dim = np.prod(row_set[:k])
        right_dim = np.prod(column_set)
    multi_index = product(*multi_index_set)
    subcore_shape = [len(indices) for indices in multi_index_set]
    core = np.zeros(subcore_shape)
    for i in multi_index:
        core[i] = F(i)
    core = reshape(core, (left_dim, dims[k], right_dim))
    return core
"""


def subcore(A, rows_array, columns_array, k):
    from itertools import chain
    rows, columns = rows_array[k], columns_array[k]
    m, n = len(rows), len(columns)

    # prepare row indices
    if k == 0:
        I = ()
    else:
        rows_list_wide = np.repeat(rows, k*n)
        I = np.unravel_index(rows_list_wide, A.shape[:k])

    # then prepare column indices
    if k == len(A.shape) - 1:
        J = ()
    else:
        columns_list_wide = np.tile(columns, m*k)
        J = np.unravel_index(columns_list_wide, A.shape[k+1:])

    # and then prepare middle index
    M = (np.tile(np.repeat(np.arange(k), n), m),)

    # We want to index A like A[I, middle, J]
    multi_index = tuple(chain(I, M, J))
    return A[multi_index]


def low_rank_approx(F, dims, r=None, delta=1e-6):
    if r == None:
        r = np.ones(len(dims) - 1)
    tt = TensorTrain
    tt.d = len(dims)
    tt.n = dims

    raw_column_indices = [np.arange(r_el) for r_el in r]
    nested_column_index_sets = [np.unravel_index(raw_column_indices[j], dims[j:]) for j in xrange(len(dims) - 1)]
    J = [np.arange(rk) for rk in r]
    I = []
    cores = []

    # First iteration - maybe I must put in into for loop
    C = subcore(F, dims, I, J, dir='make_rows', k=0)
    print C.shape
    C = reshape(C, C.shape[1:])
    Q, T = qr(C)
    I.append(maxvol(Q))
    QQ = C[I, :]
    cores.append(np.dot(Q, inv(QQ)))

    for k in xrange(len(dims) - 1):
        # take subcore C by row-column indices
        # TODO make proper subcore function
        C = subcore(F, dims, I, J, dir='make_rows', k=k)
        # and then, compute QR of C, QQ - maxvol submatrix of Q, and then compute C_k as Q QQ^-1
        Q, T = qr(C)
        I.append(maxvol(Q))
        QQ = C[I, :]
        cores.append(np.dot(Q, inv(QQ)))


def index_build(indices, sub_index, rank, k):
    if k == 0:
        indices.append(sub_index)
        return
    # unraveled indices for ranks[k] rows
    index_prev = indices[-1]

    # unravel our rn - index
    rank_rows, mode_rows = np.unravel_index(sub_index)

    new_rows = np.zeros((index_prev.size + 1, rank))
    new_rows[:-2] = index_prev[:-1]
    # fill out indices
    new_rows[:-1, :], new_rows[-1] = index_prev[:, rank_rows], mode_rows
    indices.append(new_rows)
    return


def skeleton_decomposition(A, ranks=None, eps=1e-9):
    n = A.shape
    d = len(n)
    # if ranks is not specified, define them as (2, 2, ..., 2)
    if ranks == None:
        ranks = np.ones(d + 1) * 2
        ranks[0] = ranks[-1] = 1

    J = [np.arange(rk) for rk in ranks[1:-1]]
    I = [[]]
    cores = []
    ind = np.zeros()

    # Forward iteration - using set J of columns we build set I of rows with max volume
    for k in xrange(0, d - 1):
        C = subcore(A, I, J, k)
        print C.shape

        # if k == 0, then ranks[k] is J[0] columns count, and we have ranks[0] x n_1 x ranks[1] matrix
        # otherwise C reshapes as (ranks[k] * n[k]) x ranks[k + 1] matrix
        C = reshape(C, (ranks[k] * n[k], ranks[k+1]))

        # compute QR of C, QQ - maxvol submatrix of Q, and then compute C_k as Q QQ^-1
        Q, T = qr(C)
        index_build(I, maxvol(Q), ranks[k], k)
        I.append(maxvol(Q))
        QQ = C[I, :]

        # compute next core
        cores.append(np.dot(Q, inv(QQ)).reshape((ranks[k], n[k], ranks[k+1])))
    # And we have one core left
    cores.append(subcore(A, I, J, d-1).reshape((ranks[-2], n[-1], ranks[-1])))


    I2 = I
    J2 = [[]]
    cores2 = []

    # Backward iteration - we use set I that we construct to recalculate set J
    for k in xrange(d-1, 1):
        C = subcore(A, I2, J2, k)
        print C.shape

        C = reshape(C, (ranks[k], (n[k] * ranks[k+1])))

        Q, T = qr(C)
        J.append(maxvol(Q))
