import numpy as np

from numpy import tensordot

class TensorTrain(object):
    def __init__(self, data=None, sizes=None, eps=1e-9):
        # From full array
        if isinstance(data, np.ndarray):
            A = np.asarray(data)
            self.dtype = A.dtype
            self.tt_svd(A, eps)
        # From another TT - copy constructor
        if isinstance(data, TensorTrain):
            self.dtype = data.dtype
            self.n = data.n
            self.d = data.d
            self.cores = [core.copy() for core in data.cores]
            self.r = data.r
        # Empty constructor
        elif data is None and sizes is None:
            self.dtype = np.float
            self.n = ()
            self.cores = []
            self.d = 0
            self.t = np.array([])

    def tt_svd(self, A, eps=1e-9):
        from tt_svd import tt_svd
        tt_svd(self, A, eps)

    def full_tensor(self):
        result = self.cores[0]
        for k in xrange(1, self.d):
            result = tensordot(result, self.cores[k], [len(result.shape) - 1, 0])
        return result.reshape(result.shape[1:-1])

    def tt_round(self, eps=1e-9):
        from tt_round import tt_round
        return tt_round(self, eps)

    def __neg__(self):
        from tt_basic_algebra import tt_negate
        return tt_negate(self)

    def __add__(self, other):
        from tt_basic_algebra import tt_add
        if isinstance(other, TensorTrain):
            return tt_add(self, other)
        elif isinstance(other, np.ndarray):
            return tt_add(self, TensorTrain(other))

    def __sub__(self, other):
        from tt_basic_algebra import tt_sub
        return tt_sub(self, other)

    def __mul__(self, other):
        from tt_basic_algebra import tt_hadamard_prod, tt_mul_scalar
        if isinstance(other, TensorTrain):
            return tt_hadamard_prod(self, other)
        elif isinstance(other, np.ndarray):
            return tt_hadamard_prod(self, TensorTrain(other))
        else:
            try:
                scalar = self.dtype.type(other)
                return tt_mul_scalar(self, scalar)
            except ValueError:
                print("Value must be tensor, ndarray or a scalar type")

    def __rmul__(self, other):
        from tt_basic_algebra import tt_hadamard_prod, tt_mul_scalar
        if isinstance(other, TensorTrain):
            return tt_hadamard_prod(other, self)
        elif isinstance(other, np.ndarray):
            return tt_hadamard_prod(TensorTrain(other), self)
        else:
            try:
                scalar = self.dtype.type(other)
                return tt_mul_scalar(self, scalar, rmul=True)
            except ValueError:
                print("Value must be tensor, ndarray or a scalar type")

    def tt_convolve(self, U):
        from tt_basic_algebra import tt_convolve
        return tt_convolve(self, U)

    def tt_scalar_product(self, other):
        from tt_basic_algebra import tt_scalar_product
        return tt_scalar_product(self, other)

    def printable_basic_info(self):
        return 'core elements | core shapes | suppression ranks:\n' +\
            str([cr.size for cr in self.cores]) + ' | ' + \
            str([cr.shape for cr in self.cores]) + ' | ' + \
            str([rk for rk in self.r])