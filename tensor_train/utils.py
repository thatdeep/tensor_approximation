import numpy as np

from core import TensorTrain


def frobenius_norm(A):
    if isinstance(A, TensorTrain):
        return A.tt_frobenius_norm()
    if isinstance(A, np.ndarray):
        return np.linalg.norm(A)

def rank_chop(s_values, delta):
    error = 0
    i = 1
    while i <= s_values.size and error + s_values[-i] < delta:
        error += s_values[-i]
        i += 1
    return s_values.size - i + 1