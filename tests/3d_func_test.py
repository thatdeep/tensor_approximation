import numpy as np

# TensorTrain class, TensorTrain(data)
# Uses SVD, so probably won't have low ranks
from tensor_train import TensorTrain

# Skeleton decomposition, usage: skeleton(A), A: np.ndarray
from tensor_train import skeleton

def best_func_ever(X, Y):
    coef = 3
    new_x1, new_y1 = (X[0] + 4) / coef, (Y[1] + 4) / coef
    new_x2, new_y2 = (X[0] - 4) / coef, (X[1] - 4) / coef
    return np.exp(-new_x1**2 - new_y1 ** 2)*30 + np.exp(-new_x2**2 - new_y2 ** 2)*30 + X[0] + Y[1] + 1

x, y = np.linspace(-10, 10, 100, endpoint=False)



