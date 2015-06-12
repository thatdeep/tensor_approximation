__author__ = 'Const'

import numpy as np

from index_rc import IndexRC
from numpy import dot, reshape
from numpy.linalg import qr, inv
from tensor_train import maxvol, TensorTrain


def subcore(A, irc, k, direction='lr'):
    multi_index = irc.multi_index(k, direction)
    return A[multi_index]


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


def low_rank_approx(F, dims, r=None, delta=1e-6):
    if r == None:
        r = np.ones(len(dims) - 1)
    tt = TensorTrain
    tt.d = len(dims)
    tt.n = dims

    raw_column_indices = [np.arange(r_el) for r_el in r]
    nested_column_index_sets = [np.unravel_index(raw_column_indices[j], dims[j:]) for j in xrange(len(dims) - 1)]
    J = [np.arange(rk) for rk in r]
    J = []
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


def index_set_iteration(A, irc, direction='lr'):
    left_right, right_left = ['lr', 'LR'], ['rl', 'RL']
    if direction not in (left_right + right_left):
        raise Exception("direction must be 'lr' | 'LR' (left-to-right), or 'rl' | 'RL' (right-to-left)")
    if direction in left_right:
        k_range = xrange(0, irc.d - 1)
    else:
        k_range = xrange(irc.d - 1, 0, -1)

    cores = []
    for k in k_range:
        C = subcore(A, irc, k, direction=direction)
        #print C.shape

        if direction in left_right:
            C = reshape(C, (irc.ranks[k] * irc.n[k], irc.ranks[k+1]))
        else:
            C = reshape(C, (irc.ranks[k], (irc.n[k] * irc.ranks[k+1]))).T

        # compute QR of C, QQ - maxvol submatrix of Q, and then compute C_k as Q QQ^-1
        Q, T = qr(C)
        rows = maxvol(Q)
        irc.update_index(rows, k, direction=direction)
        QQ = Q[rows, :]

        # compute next core
        next_core = np.dot(Q, inv(QQ))
        if direction in right_left:
            next_core = next_core.T
        cores.append(next_core.reshape((irc.ranks[k], irc.n[k], irc.ranks[k+1])))
    if direction in left_right:
        cores.append(subcore(A, irc, irc.d - 1).reshape((irc.ranks[-2], irc.n[-1], irc.ranks[-1])))
    else:
        cores.append(subcore(A, irc, 0).reshape(irc.ranks[0], irc.n[0], irc.ranks[1]))
        cores = cores[::-1]
    return cores


def skeleton(A, ranks=None, eps=1e-9):
    n = A.shape
    d = len(n)
    # if ranks is not specified, define them as (2, 2, ..., 2)
    if ranks == None:
        ranks = np.ones(d + 1, dtype=int) * 2
        ranks[0] = ranks[-1] = 1

    irc = IndexRC(n, ranks[:])

    cores = index_set_iteration(A, irc, direction='lr')
    cores2 = index_set_iteration(A, irc, direction='rl')

    return cores, cores2



def skeleton_decomposition_old(A, ranks=None, eps=1e-9):
    n = A.shape
    d = len(n)
    # if ranks is not specified, define them as (2, 2, ..., 2)
    if ranks == None:
        ranks = np.ones(d + 1, dtype=int) * 2
        ranks[0] = ranks[-1] = 1

    irc = IndexRC(n, ranks[:])
    cores = []

    # Forward iteration - using set J of columns we build set I of rows
    # that corresponds to submatrix of max volume
    for k in xrange(0, d - 1):
        C = subcore(A, irc, k)
        #print C.shape

        # if k == 0, then ranks[k] is J[0] columns count, and we have ranks[0] n_1 x ranks[1] matrix
        # otherwise C reshapes as (ranks[k] * n[k]) x ranks[k + 1] matrix
        C = reshape(C, (ranks[k] * n[k], ranks[k+1]))

        # compute QR of C, QQ - maxvol submatrix of Q, and then compute C_k as Q QQ^-1
        Q, T = qr(C)
        rows = maxvol(Q)
        irc.update_index(rows, k)
        QQ = Q[rows, :]

        # compute next core
        cores.append(np.dot(Q, inv(QQ)).reshape((ranks[k], n[k], ranks[k+1])))
    # And we have one core left
    cores.append(subcore(A, irc, d-1).reshape((ranks[-2], n[-1], ranks[-1])))

    cores2 = []

    # Backward iteration - we use set I that we construct to recalculate set J
    for k in xrange(d-1, 0, -1):
        C = subcore(A, irc, k, direction='rl')
        #print C.shape

        C = reshape(C, (ranks[k], (n[k] * ranks[k+1]))).T

        Q, T = qr(C)
        columns = maxvol(Q)
        irc.update_index(columns, k, direction='rl')
        # not sure
        QQ = Q[columns, :]

        # compute next core
        cores2.append(np.dot(Q, inv(QQ)).T.reshape((ranks[k], n[k], ranks[k+1])))

    cores2.append(subcore(A, irc, 0).reshape(ranks[0], n[0], ranks[1]))
    cores2 = cores2[::-1]

    return cores, cores2
