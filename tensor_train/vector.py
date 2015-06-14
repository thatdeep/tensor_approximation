import math
import numpy as np

from numpy import reshape
from core import TensorTrain


def unraveled_index(i, d):
    bin_repr = np.zeros(d)
    repr_tail = np.array([int(i_bit) for i_bit in bin(i)[2:]])
    bin_repr[d - repr_tail.size :] = repr_tail
    return bin_repr

def is_pow_two(number):
  return (number & (number-1) == 0) and (number != 0)





class Vector(TensorTrain):
    def __init__(self, black_box):
        self.black_box = black_box
        TensorTrain.__init__(self, )

    def from_array(self, data, d):
        data = np.ravel(np.asarray(data))
        if not is_pow_two(data.size):
            raise Exception('array size must be power of two, but it have {n} elements!'.format(n=data.size))
        if d == None:
            d = int(math.log(data.size, 2))
        if data.size != 2**d:
            raise Exception('there must be 2**d == data.size!')

        new_shape = tuple([2]*d)
        data_reshaped = reshape(data, new_shape)
        black_box =  BlackBox(lambda i: data_reshaped[i], new_shape, d)
        self.__init__(black_box)
