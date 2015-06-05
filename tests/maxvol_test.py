import numpy as np

from tensor_train import maxvol
from numpy.linalg import det
from itertools import combinations


def maxvol_test(A):
    n, r = A.shape
    ind = maxvol(A)
    am_det = det(A[ind, :])
    print ind
    dets = []
    for row_combo in combinations(range(n), r):
        dets.append((row_combo, det(A[row_combo, :])))
    return [d for d in dets if abs(d[1]) > abs(am_det)], am_det