import math
import numpy as np
import mpmath as mp
from mpmath import workdps, mpf

from numpy import tensordot
from black_box import BlackBox

class TensorTrain(object):
    """TensorTrain class - implementation of TT format.

    TT format include representation of tensor as a convolution of
    low-rank tensors (cores) with suppression ranks.

    Attributes:
        n (tuple): Shape of tensor
        d (int): Number of dimensions
        r (np.ndarray): Compression ranks
        cores (list[np.ndarray]): TT cores
        dtype (np.dtype): Data type of tensor values


    """
    def __init__(self, black_box=None, decompose='skeleton', eps=1e-9):
        from skeleton import skeleton
        accepted_types = ['svd', 'skeleton']
        if decompose not in accepted_types:
            raise Exception('decompose must be one of the following: {types}'.format(types=accepted_types))

        if black_box == None:
            # Empty constructor
            self.dtype = np.float
            self.n = ()
            self.shape = ()
            self.cores = []
            self.d = 0
            self.r = np.array([])
            self.norm = 0
            self._normed = True
        elif isinstance(black_box, TensorTrain):
            # Copy constructor
            self.dtype = black_box.dtype
            self.n = black_box.n
            self.shape = ()
            self.d = black_box.d
            self.r = black_box.r
            if black_box.normed():
                self.norm = black_box.norm
                self._normed = True
            else:
                self._normed = False
            self.cores = [core.copy() for core in black_box.cores]
        elif isinstance(black_box, BlackBox):
            # Constructor from some "Black Box" tensor from which we can
            # retrieve arbitrary elements
            self.dtype = black_box.dtype
            self.n = black_box.n
            self.shape = black_box.shape
            self.d = black_box.d
            if decompose == 'skeleton':
                self.__init__(skeleton(black_box, cores_only=False, eps=eps), eps=eps)
                #self.cores = skeleton(black_box, cores_only=True, eps=eps)
                #self.r = np.array([1] + [core.shape[2] for core in self.cores])
            elif decompose == 'svd':
                self.tt_svd(black_box, eps=eps)

    @classmethod
    def from_array(cls, data, decompose='skeleton'):
        accepted_types = ['svd', 'skeleton']
        if decompose not in accepted_types:
            raise Exception('decompose must be one of the following: {types}'.format(types=accepted_types))
        if isinstance(data, np.ndarray):
            black_box = BlackBox.from_array(data)
            cls(black_box, decompose)

    @classmethod
    def from_cores(cls, cores, reuse=False, norm=None):
        dps=500
        tt = TensorTrain()
        if norm is not None:
            with workdps(dps):
                norm = mpf(norm)
                tt.norm = norm
                tt._normed = True
        else:
            tt._normed = False
        if len(cores) == 0:
            return tt
        tt.dtype = cores[0].dtype
        tt.n = tuple([core.shape[1] for core in cores])
        tt.r = np.array([1] + [core.shape[2] for core in cores])
        tt.d = len(cores)
        tt.cores = [core for core in cores] if reuse else [core.copy() for core in cores]
        return tt

    def normed(self):
        return self._normed

    def __getitem__(self, item):
        it = np.asarray(item)
        if len(it.shape) == 1:
            subcores = [self.cores[i][:, ii, :] for i, ii in enumerate(it)]
            v = subcores[0]
            for i in xrange(1, self.d):
                v = np.dot(v, subcores[i])
            # b must be [1, 1] matrix
            return v[0, 0]


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

    def tt_outer_product(self, other):
        from tt_basic_algebra import tt_outer_product
        return tt_outer_product(self, other)

    def tt_frobenius_norm(self):
        val = self.tt_scalar_product(self)
        return np.sqrt(val)

    def printable_basic_info(self):
        return 'core elements | core shapes | suppression ranks:\n' +\
            str([cr.size for cr in self.cores]) + ' | ' + \
            str([cr.shape for cr in self.cores]) + ' | ' + \
            str([rk for rk in self.r])