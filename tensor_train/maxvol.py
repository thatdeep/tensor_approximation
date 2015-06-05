__author__ = 'Const'

import numpy as np

from scipy.linalg import lu


# An alternative to MATLAB '/' matrix operation
def rmatdiv(B, A):
    assert A.shape[1] == B.shape[1]
    z = np.linalg.lstsq(A.T, B.T)[0].T
    return z

def maxvol(A, ind=None):
    n, r = A.shape
    if n <= r:
        return np.array(xrange(n))

    if ind == None:
        perm, l_temp, u_temp = lu(A)
        p = perm.T.nonzero()[1].copy()
        ind = p[:r]
        B = A[p, :]
        submatrix = A[ind, :]
        z = rmatdiv(B[r:, :], submatrix)

    all_iters = 100
    eps = 5e-2
    iter = 0
    while iter <= all_iters:
        max_indices = np.argmax(abs(z), axis=1)
        max_values = abs(z)[range(z.shape[0]), max_indices]
        max_arg = np.argmax(max_values)
        max_val = max_values[max_arg]
        if max_val <= 1 + eps:
            ind = np.sort(ind)
            return ind
        i_row = max_arg
        j_row = max_indices[i_row]
        p[i_row + r], p[j_row] = p[j_row], p[i_row + r]
        bb = z[:, j_row]
        bb[i_row] += 1
        cc = z[i_row, :]
        cc[j_row] -= 1
        print z.shape, bb.shape, cc.shape
        z -= np.outer(bb, cc) / z[i_row, j_row]
        iter += 1
        ind = p[:r]
        ind = np.sort(ind)
    return ind