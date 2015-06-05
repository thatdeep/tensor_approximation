import numpy as np
import math

from numpy import dot, zeros, ones, eye, identity as I, reshape, tensordot, kron
from numpy.linalg import svd, norm, qr

from core import TensorTrain


def tt_negate(tt):
    A = TensorTrain()
    A.d = tt.d
    A.n = tt.n
    A.r = tt.r
    A.cores = tt.cores
    A.cores[0] = -A.cores[0]
    return A


def tt_mul_scalar(tt, scalar, rmul=False):
    A = TensorTrain(tt)
    if scalar != 0:
        if rmul:
            A.cores[-1] *= scalar
        else:
            A.cores[0] *= scalar
    else:
        for k in xrange(A.cores):
            A.cores[k] = np.zeros((A.n[k]))
            A.cores[k] = reshape(A.cores[k], (1, A.n[k], 1))
    return A


def tt_add(tt, other):
    A = TensorTrain()
    if tt.n != other.n:
        raise Exception('tensor dimensions ' + str(tt.n) + ' and ' + str(other.n) + "didn't match!")
    A.d = tt.d
    A.n = tt.n
    A.r = tt.r + other.r
    if tt.r[0] == other.r[0]:
        A.r[0] = tt.r[0]
    else:
        raise Exception("First ranks must match!")
    if tt.r[-1] == other.r[-1]:
        A.r[-1] = tt.r[-1]
    else:
        raise Exception("Last ranks must match!")

    A.cores = []

    next_core = np.zeros((A.r[0], A.n[0], A.r[1]))
    next_core[:, :, :tt.r[1]] = tt.cores[0]
    next_core[:, :, tt.r[1]:] = other.cores[0]
    A.cores.append(next_core)

    for i in xrange(1, A.d - 1):
        next_core = zeros((A.r[i], A.n[i], A.r[i+1]))
        next_core[:tt.r[i], :, :tt.r[i+1]] = tt.cores[i]
        next_core[tt.r[i]:, :, tt.r[i+1]:] = other.cores[i]
        A.cores.append(next_core)

    next_core = np.zeros((A.r[-2], A.n[-1], A.r[-1]))
    next_core[:tt.r[-2], :, :] = tt.cores[-1]
    next_core[tt.r[-2]:, :, :] = other.cores[-1]
    A.cores.append(next_core)
    return A


def tt_sub(tt, other):
    return tt + -other


def tt_convolve(tt, U):
    if len(U) != tt.d:
        raise Exception('Tensor of rank d must convolve with d vectors!')
    if not all([a == b for a, b in zip(tt.n, [u.size for u in U])]):
        raise Exception('Tensor must convolve with vectors of proper dimensions!')
    v = 1
    for k in xrange(tt.d):
        next_matrix = tensordot(tt.cores[k], U[k], [1, 0])
        v = dot(v, next_matrix)
    # And then, we have an imitative ranks of 1 from left and right, so erase them
    return v[0, 0]


def tt_hadamard_prod(tt, other):
    A = TensorTrain()
    if tt.n != other.n:
        raise Exception('tensor dimensions ' + str(tt.n) + ' and ' + str(other.n) + "didn't match!")
    A.d = tt.d
    A.n = tt.n
    A.r = tt.r * other.r

    A.cores = []

    for k in xrange(A.d):
        self_r = tt.r[k]
        self_r1 = tt.r[k+1]
        other_r = other.r[k]
        other_r1 = other.r[k+1]
        next_core = np.zeros((self_r * other_r, tt.n[k], self_r1 * other_r1))
        for i in xrange(tt.n[k]):
            next_core[:, i, :] = kron(tt.cores[k][:, i, :], other.cores[k][:, i, :])
        A.cores.append(next_core)
    return A


def tt_scalar_product(tt, other):
    A = tt * other
    return A.tt_convolve([np.ones(dimension) for dimension in A.n])