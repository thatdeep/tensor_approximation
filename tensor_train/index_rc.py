import numpy as np

from numpy import reshape


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