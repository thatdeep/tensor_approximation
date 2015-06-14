import numpy as np

from numpy import reshape
from maxvol import maxvol
from index_rc import IndexRC

from numpy.linalg import inv, qr
from core import TensorTrain
from utils import frobenius_norm


def subcore(A, irc, k, direction='lr'):
    multi_index = irc.multi_index(k, direction)
    return A[multi_index]

def index_set_iteration(A, irc, direction='lr', explicit_cores=True):
    if type(explicit_cores) is not bool:
        raise Exception('explicit_cores flag must be True of False')

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

        # compute next core and append cores only if we need explicit cores
        if explicit_cores:
            QQ = Q[rows, :]

            # compute next core
            next_core = np.dot(Q, inv(QQ))
            if direction in right_left:
                next_core = next_core.T

            cores.append(next_core.reshape((irc.ranks[k], irc.n[k], irc.ranks[k+1])))
    # compute last core if we need explicit cores
    if explicit_cores:
        if direction in left_right:
            cores.append(subcore(A, irc, irc.d - 1).reshape((irc.ranks[-2], irc.n[-1], irc.ranks[-1])))
        else:
            cores.append(subcore(A, irc, 0).reshape(irc.ranks[0], irc.n[0], irc.ranks[1]))
            cores = cores[::-1]
        return cores


def recalculate_ranks(ranks, rounded_ranks, unstable_ranks, maxranks):
    unstable_indices = np.where(unstable_ranks)
    stabilization_indices = np.where(np.logical_or(ranks > rounded_ranks, ranks >= maxranks))
    unstable_ranks[stabilization_indices] = False
    ranks[unstable_ranks] += 1

def skeleton(A, ranks=None, cores_only=False, eps=1e-6, max_iter=10):
    n = A.shape
    d = len(n)
    # if ranks is not specified, define them as (2, 2, ..., 2)
    maxranks = np.asarray(n)[1:]
    maxranks[np.asarray(n[:-1]) < maxranks] = np.asarray(n[:-1])[np.asarray(n[:-1]) < maxranks]
    maxranks = np.array([1] + list(maxranks) + [1])
    if ranks == None:
        ranks = np.ones(d + 1, dtype=int) * 2
        ranks[0] = ranks[-1] = 1
    ranks_unstable = np.ones_like(ranks, dtype=np.bool)
    ranks_unstable[0] = ranks_unstable[-1] = False

    while True:
        irc = IndexRC(n, ranks[:])

        # perform first approximation
        index_set_iteration(A, irc, direction='lr', explicit_cores=False)
        prev_approx = TensorTrain.from_cores(index_set_iteration(A, irc, direction='rl'))
        print "Inner iterations with ranks {r}".format(r=ranks)
        for i in xrange(max_iter):
            # perform next approximation
            index_set_iteration(A, irc, direction='lr', explicit_cores=False)
            next_approx = TensorTrain.from_cores(index_set_iteration(A, irc, direction='rl'))

            # TODO Did we really need tt_round here?
            difference = frobenius_norm((next_approx - prev_approx).full_tensor())
            print "difference: {d}, eps: {eps}".format(d=difference, eps=eps)
            if difference < eps:
                print "Reach close approximation on {i} iteration with ranks {r}".format(i=i+1, r=ranks)
                break
            prev_approx = next_approx

        # now we have approximation to tensor A with fixed ranks
        rounded_approx = next_approx.tt_round()
        rounded_ranks = rounded_approx.r
        print "ranks    : {r}\nnew ranks: {nr}".format(r=ranks, nr=rounded_ranks)
        recalculate_ranks(ranks, rounded_ranks, ranks_unstable, maxranks)
        if not np.any(ranks_unstable):
            # All ranks are stablilize
            print "Stabilize!"
            return rounded_approx.cores if cores_only else rounded_approx