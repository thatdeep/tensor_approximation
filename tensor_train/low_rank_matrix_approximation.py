__author__ = 'Const'

import numpy as np

from numpy.linalg import qr
from tensor_train import maxvol


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
