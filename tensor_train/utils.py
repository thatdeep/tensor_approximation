from numpy.linalg import norm
from numpy import reshape


def frobenius_norm(A):
    return norm(reshape(A, A.size))


def rank_chop(s_values, delta):
    error = 0
    i = 1
    while i <= s_values.size and error + s_values[-i] < delta:
        error += s_values[-i]
        i += 1
    return s_values.size - i + 1