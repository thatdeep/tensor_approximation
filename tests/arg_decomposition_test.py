import numpy as np

"""
def arg_approximation_delta(x):
    x = np.random.random() * 100
    sign, bits = x_decomposition(x)
    xx = x_restore(sign, bits)
    return abs(x - xx)


def arg_vector_approximation_test(v, eps=1e-15):
    assert np.array([arg_approximation_delta(x) for x in v]).max() < eps


def uniformly_distributed_approximation_test(size, bound):
    arg_vector_approximation_test((np.random.random(size) - 0.5) * 2 * bound)


def small_decomposition_test(x_small=1e-500):
    assert arg_approximation_delta(x_small) == 0
    """