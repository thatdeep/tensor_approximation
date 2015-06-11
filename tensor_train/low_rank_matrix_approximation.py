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


def full_index(row_part, column_part, mid_size):
    from itertools import chain

    if type(row_part) == np.ndarray:
        if len(row_part.shape) == 1:
            m = row_part.size if row_part.size else 1
            di = 1 if row_part.size else 0
        else:
            m = row_part.shape[1]
            di = row_part.shape[0]
    else:
        m = 1

    if type(column_part) == np.ndarray:
        if len(column_part.shape) == 1:
            n = column_part.size if column_part.size else 1
            dj = 1 if column_part.size else 0
        else:
            n = column_part.shape[1]
            dj = column_part.shape[0]
    else:
        n = 1

    #m = 1 if type(row_part) == np.ndarray and len(row_part.shape) == 1 else row_part.shape[0]
    #n = 1 if type(column_part) == np.ndarray and len(column_part.shape) == 1 else column_part.shape[0]
    mid = mid_size

    I = np.repeat(row_part, mid * n) if row_part.shape else ()
    if di != 0:
        I = reshape(I, (di, -1))
    M = np.tile(np.repeat(np.arange(mid), n), m)
    J = np.tile(column_part, m * mid) if column_part.shape else ()
    if dj != 0:
        J = reshape(J, (dj, -1))
    print I, M, J
    multi_index = tuple(chain(tuple(I), (M,), tuple(J)))
    return multi_index


def subcore(A, irc, k, direction='lr'):
    multi_index = irc.multi_index(k, direction)
    return A[multi_index]



def index_build(indices, sub_index, rank, dim):
    if not indices:
        indices.append(sub_index)
        return
    # unraveled indices for ranks[k] rows
    index_prev = indices[-1]
    density = 1 if len(index_prev.shape) == 1 else index_prev.shape[0]
    # unravel our rn - index
    rank_rows, mode_rows = np.unravel_index(sub_index, (rank, dim))

    new_rows = np.zeros((density + 1, rank), dtype=int)
    #new_rows[:-2] = index_prev[:-1]
    # fill out indices
    new_rows[:-1, :], new_rows[-1] = index_prev[:, rank_rows], mode_rows
    indices.append(new_rows)
    return


class IndexRC(object):
    def __init__(self, n, ranks, initial_index=None):
        self.n = n
        self.d = len(n)

        if hasattr(ranks, "__len__"):
            self.ranks = ranks
        else:
            self.ranks = np.ones(self.d + 1, dtype=int)*ranks
            ranks[0] = ranks[-1] = 1

        if initial_index == None:
            self.index = [0]*(self.d - 1)
            nd = np.array(self.n)
            steps = np.array(nd[:-1], dtype=int)
            steps[nd[1:] < steps] = nd[1:][nd[1:] < steps]
            steps /= ranks[1:-1]
            assert not np.sum(steps == 0)

            self.update_index(np.arange(0, ranks[1]*steps[0], step=steps[0], dtype=int), k=0)
            for k in xrange(1, self.d-1, 1):
                self.update_index(np.arange(0, ranks[k + 1]*steps[k], step=steps[k], dtype=int), k)
            self.index = self.index[::-1]
        else:
            self.index = initial_index

        print self.d - 1, len(self.index)
        assert self.d - 1 == len(self.index), 'Mode number and size of initial index must be equal'


    def update_index(self, sub_index, k, direction='lr'):
        if direction == 'lr' or direction == 'LR':
            if k == 0:
                self.index[0] = sub_index
                return
            rank_rows, mode_rows = np.unravel_index(sub_index, (self.ranks[k], self.n[k]))
            if k == 1:
                self.index[1] = np.vstack([self.index[0][rank_rows], mode_rows])
                return
            density = self.index[k - 1].shape[0]
            self.index[k] = np.zeros((density + 1, self.ranks[k + 1]), dtype=int)
            self.index[k][:-1] = self.index[k - 1][:, rank_rows]
            self.index[k][-1] = mode_rows
        if direction == 'rl' or direction == 'RL':
            k -= 1
            if k == self.d - 2:
                self.index[-1] = sub_index
                return
            mode_columns, rank_columns = np.unravel_index(sub_index, (self.n[k + 1], self.ranks[k + 2]))
            if k == self.d - 3:
                self.index[k] = np.vstack([mode_columns, self.index[-1][rank_columns]])
                return
            density = self.index[k + 1].shape[0]
            self.index[k] = np.zeros((density + 1, self.ranks[k + 1]), dtype=int)
            self.index[k][0] = mode_columns
            self.index[k][1:] = self.index[k + 1][:, rank_columns]
        return

    def __getitem__(self, item):
        return self.index[item]

    def multi_index(self, k, direction='lr'):
        left_directions, right_directions = ['lr', 'LR'], ['rl', 'RL']
        if direction in right_directions:
            k -= 1
        if direction in (left_directions + right_directions):
            mid = self.n[k]
            I = self.index[k - 1] if k > 0 else np.array([])
            J = self.index[k] if k < self.d - 1 else np.array([])
            multi_index = full_index(I, J, mid)
            return multi_index
        else:
            raise Exception("Wrong direction parameter (Must be 'lr' or 'rl')")


def skeleton_decomposition(A, ranks=None, eps=1e-9):
    n = A.shape
    d = len(n)
    # if ranks is not specified, define them as (2, 2, ..., 2)
    if ranks == None:
        ranks = np.ones(d + 1, dtype=int) * 2
        ranks[0] = ranks[-1] = 1

    irc = IndexRC(n, ranks[:])
    cores = []

    # Forward iteration - using set J of columns we build set I of rows with max volume
    for k in xrange(0, d - 1):
        C = subcore(A, irc, k)
        print C.shape

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
        C = subcore(A, irc, k)
        print C.shape

        C = reshape(C, (ranks[k], (n[k] * ranks[k+1]))).T

        Q, T = qr(C)
        columns = maxvol(Q)
        irc.update_index(columns, k, direction='rl')
        # not sure
        QQ = Q[columns, :]

        # compute next core
        cores2.append(np.dot(Q, inv(QQ)).T.reshape((ranks[k], n[k], ranks[k+1])))

    print 11

    cores2.append(subcore(A, irc, 0).reshape(ranks[0], n[0], ranks[1]))
    cores2 = cores2[::-1]

    return cores, cores2
