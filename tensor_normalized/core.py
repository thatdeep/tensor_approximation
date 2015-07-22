import math
import numpy as np
import mpmath as mp
from mpmath import workdps, mpf

from numpy import tensordot
from black_box import BlackBox

from constants import TENSOR_NORM_DPS

class Tensor(object):
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
    def __init__(self, black_box=None, decompose='svd', eps=1e-9):
        accepted_types = ['svd', 'skeleton']
        if decompose not in accepted_types:
            raise Exception('decompose must be one of the following: {types}'.format(types=accepted_types))

        if black_box == None:
            # Empty constructor
            self.dtype = np.float
            self.shape = ()
            self.cores = []
            self.d = 0
            self.r = np.array([])
            with workdps(TENSOR_NORM_DPS):
                self.norm = mpf('0')
        elif isinstance(black_box, np.ndarray):
            self.dtype = black_box.dtype
            self.shape = black_box.shape
            self.d = len(self.n)
            if decompose == 'skeleton':
                pass
                #self.__init__(skeleton(black_box, cores_only=False, eps=eps), eps=eps)
            elif decompose == 'svd':
                self.tt_svd(black_box, eps=eps)
        elif isinstance(black_box, Tensor):
            # Copy constructor
            self.dtype = black_box.dtype
            self.shape = black_box.shape
            self.d = black_box.d
            self.r = black_box.r
            self.cores = [core.copy() for core in black_box.cores]
            with workdps(TENSOR_NORM_DPS):
                self.norm = mpf(black_box.norm)
        elif isinstance(black_box, BlackBox):
            # Constructor from some "Black Box" tensor from which we can
            # retrieve arbitrary elements
            self.dtype = black_box.dtype
            self.shape = black_box.shape
            self.d = black_box.d
            if decompose == 'skeleton':
                pass
                #self.__init__(skeleton(black_box, cores_only=False, eps=eps), eps=eps)
                #self.cores = skeleton(black_box, cores_only=True, eps=eps)
                #self.r = np.array([1] + [core.shape[2] for core in self.cores])
            elif decompose == 'svd':
                self.tt_svd(black_box, eps=eps)

    @classmethod
    def from_array(cls, data, eps=1e-7, decompose='skeleton'):
        accepted_types = ['svd', 'skeleton']
        if decompose not in accepted_types:
            raise Exception('decompose must be one of the following: {types}'.format(types=accepted_types))
        if isinstance(data, np.ndarray):
            return cls(data, decompose, eps)

    @classmethod
    def from_cores(cls, cores, reuse=False):
        dps=500
        tt = Tensor()
        if len(cores) == 0:
            return tt
        tt.dtype = cores[0].dtype
        tt.n = tuple([core.shape[1] for core in cores])
        tt.r = np.array([1] + [core.shape[2] for core in cores])
        tt.d = len(cores)
        tt.cores = [core for core in cores] if reuse else [core.copy() for core in cores]
        return tt

    def __getitem__(self, item):
        it = np.asarray(item)
        if len(it.shape) == 1:
            subcores = [self.cores[i][:, ii, :] for i, ii in enumerate(it)]
            v = subcores[0]
            for i in xrange(1, self.d):
                v = np.dot(v, subcores[i])
            # b must be [1, 1] matrix
            return v[0, 0] * np.float128(self.norm)


    def tt_svd(self, A, eps=1e-9):
        from tt_svd import tt_svd
        tt_svd(self, A, eps)

    def full_tensor(self):
        result = self.cores[0]
        for k in xrange(1, self.d):
            result = tensordot(result, self.cores[k], [len(result.shape) - 1, 0])
        return result.reshape(result.shape[1:-1]) * np.float128(self.norm)

    #def tt_round(self, eps=1e-9):
    #    from tt_round import tt_round
    #    return tt_round(self, eps)

    def tt_frobenius_norm(self):
        return np.float128(self.norm)

    def printable_basic_info(self):
        return 'core elements | core shapes | suppression ranks:\n' +\
            str([cr.size for cr in self.cores]) + ' | ' + \
            str([cr.shape for cr in self.cores]) + ' | ' + \
            str([rk for rk in self.r])