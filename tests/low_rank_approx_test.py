
import numpy as np

from tensor_train import low_rank_matrix_approx

def approx_test(A, r):
    #A = np.random.random((10, 4))
    I, J = low_rank_matrix_approx(A, r)

    print I
    print J