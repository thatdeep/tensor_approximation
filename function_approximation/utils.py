import math
import numpy as np
import mpmath as mp


def decompose(x, gamma_pow, precision=40):
    with mp.workdps(2048):
        tail = x
        compressed = np.zeros(precision + 1)
        for i in xrange(precision + 1):
            if tail == 0:
                break
            tail, compressed[i] = mp.frac(tail), mp.floor(tail)
            tail = mp.ldexp(tail, gamma_pow)
        return compressed


def strong_decompose(x, gamma, precision=40):
    tail = x
    compressed = np.zeros(precision + 1)
    for i in xrange(precision + 1):
        if tail == 0:
            break
        tail, compressed[i] = math.modf(tail)
        tail = tail * gamma
    return compressed


def rfunc_builder(gamma, compressed):
    return np.logical_or(compressed == (gamma - 1), compressed == (gamma - 2))


def x_restore(sign, bits):
    return (-1)**sign * np.sum((bits * np.array([2**(-i) for i in xrange(bits.size)])))

# returns smallest power of 2 that is greater or equal than v
def power_2_bound(v):
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    return v + 1


def gamma_estimate(N):
    #return power_2_bound(2*N + 3)
    return 2**N+2