import numpy as np

from core import Tensor


def tt_zeros(n):
    # TODO what shall we do if we have one/two dimensional tensors?
    return Tensor.from_cores([np.zeros(elements)[np.newaxis, ..., np.newaxis] for elements in n])