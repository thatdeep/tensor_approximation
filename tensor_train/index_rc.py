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
            self.ranks[0] = self.ranks[-1] = 1

        if initial_index == None:
            self.index = [0]*(self.d - 1)
            #nd = np.array(self.n)
            #steps = np.array(nd[:-1], dtype=int)
            #steps[nd[1:] < steps] = nd[1:][nd[1:] < steps]
            #steps /= ranks[1:-1]
            #assert not np.sum(steps == 0)

            for k in xrange(self.d - 1):
                self.index[k] = semi_random_multi_index(self.n[k + 1:], self.ranks[k+1])
            #self.update_index(np.arange(0, ranks[self.d - 1]*steps[0], step=steps[0], dtype=int), k=self.d - 1, direction='rl')
            #for k in xrange(self.d - 2, 0, -1):
            #    self.update_index(np.arange(0, ranks[k]*steps[k], step=steps[k], dtype=int), k, direction='rl')
            #xrange(irc.d - 1, 0, -1)
            #self.update_index(np.arange(0, ranks[1]*steps[0], step=steps[0], dtype=int), k=0)
            #for k in xrange(1, self.d-1, 1):
            #    self.update_index(np.arange(0, ranks[k + 1]*steps[k], step=steps[k], dtype=int), k)
            #self.index = self.index[::-1]
        else:
            self.index = initial_index

        #print self.d - 1, len(self.index)
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
        #if direction in right_directions:
        #    k -= 1
        if direction in (left_directions + right_directions):
            mid = self.n[k]
            I = self.index[k - 1] if k > 0 else np.array([])
            J = self.index[k] if k < self.d - 1 else np.array([])
            multi_index = full_index(I, J, mid)
            return multi_index
        else:
            raise Exception("Wrong direction parameter (Must be 'lr' or 'rl')")


def semi_random_multi_index(shape, r):
    sizeall = reduce(lambda x, y: x*y, shape, 1)
    d = len(shape)
    if r == sizeall:
        print "full ranks ranks[{r}]".format(r=r)
        return np.vstack(np.unravel_index(np.arange(sizeall, dtype=int), shape))
    if r > sizeall:
        assert False
        self.ranks[k+1] = sizeall
        print "We cut off ranks[{k}] from {r} to {r1}".format(k=k+1, r=r, r1=sizeall)
        return np.vstack(np.unravel_index(np.arange(sizeall, dtype=int), shape))
        raise Exception("You can't create {r} uniq elements from {n} elements".format(r=r, n=sizeall))
    if r > sizeall*9/10:
        print "Warning! We have very little probability for reach {r} uniqs".format(r=r)
    uniqs = set()

    # Just create some more random elements than r
    rand_indices = np.vstack([np.random.randint(0, dim_size, 2*r) for dim_size in shape]).T
    for rand_index in rand_indices:
        tupled = tuple(rand_index)
        if tupled not in uniqs:
            uniqs.add(tupled)
        if len(uniqs) == r:
            break

    # If we haven't find enough unique indices just take them with loop
    if len(uniqs) < r:
        delta = r - len(uniqs)
        print 'delta: {d}'.format(d=delta)
        while delta > 0:
            tupled = tuple([np.random.randint(0, dim_size) for dim_size in shape])
            if tupled not in uniqs:
                uniqs.add(tupled)
                delta -= 1
    # Return indices as list of np.ndarray
    return np.vstack([np.array(uniq) for uniq in uniqs]).T


def full_index(row_part, column_part, mid_size):
    from itertools import chain

    def unpack_nd(values):
        # returns number of elements by multiindex and number of dimensions
        if len(values.shape) == 0:
            # scalar value
            return 1, 1
        elif len(values.shape) == 1:
            # vector with shape (s, )
            return (values.size, 1) if values.size else (1, 0)
        elif len(values.shape) == 2:
            # stack of vectors with shape (s, t)
            return values.shape[::-1]

    row_part = np.asarray(row_part, dtype=int)
    column_part = np.asarray(column_part, dtype=int)

    m, di = unpack_nd(row_part)
    mid = mid_size
    n, dj = unpack_nd(column_part)

    sum_dim = di + 1 + dj
    cur_idx = 0
    big_screen_index = np.zeros((sum_dim, m * mid * n), dtype=int)
    if di != 0:
        big_screen_index[0:di] = np.repeat(row_part, mid * n).reshape((di, -1))
    big_screen_index[di: di + 1] = np.tile(np.repeat(np.arange(mid), n), m)
    if dj != 0:
        big_screen_index[di + 1:] =  np.tile(column_part, m * mid).reshape((dj, -1))

    #I = np.repeat(row_part, mid * n) if row_part.shape else ()
    #M = np.tile(np.repeat(np.arange(mid), n), m)
    #J = np.tile(column_part, m * mid) if column_part.shape else ()

    #if di != 0:
    #    I = reshape(I, (di, -1))
    #if dj != 0:
    #    J = reshape(J, (dj, -1))

    #print I, M, J
    #return tuple(chain(tuple(I), (M,), tuple(J)))
    return big_screen_index
