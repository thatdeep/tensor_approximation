import numpy as np
from numpy import random as rnd
from tensor_train.utils import frobenius_norm
from tensor_train.core import TensorTrain


def sparse_tensor(d=4, n=40, sparsity_percent=0.95):
    shape_form = tuple([n]*d)
    sparsity_mask = np.array(rnd.binomial(1, sparsity_percent, shape_form), dtype=np.bool)
    A = np.random.random(shape_form)
    A[sparsity_mask] = 0
    return A


def validate_tensor_operation(tt_a, tt_b, operation = lambda x, y: x + y, eps=1e-14):
    tt_c = operation(tt_a, tt_b)

    A = tt_a.full_tensor()
    B = tt_b.full_tensor()
    C = tt_c.full_tensor()

    diff_norm = np.linalg.norm(C - operation(A, B))
    return diff_norm < eps


def simple_arithmetic_test(d=4, n=5, operation='add', percent=0.95, variate_dims=False, eps=1e-10):
    operations = {
        'add': lambda x, y: x+y,
        'sub': lambda x, y: x-y,
        'mul': lambda x, y: x*y,
        'div': lambda x, y: x/y,
    }
    #and it
    sign_map = {
        'add': '+',
        'sub': '-',
        'mul': '*',
        'div': '/',
    }
    if not variate_dims:
        shape_form = tuple(n for _ in xrange(d))
    else:
        shape_form = tuple(rnd.randint(n / 2 + 1, n + n / 2, d))
    A = sparse_tensor(d, n, percent)
    B = sparse_tensor(d, n, percent)

    t_a = TensorTrain.from_array(A, eps)
    t_b = TensorTrain.from_array(B, eps)

    t_a_full = t_a.full_tensor()
    t_b_full = t_b.full_tensor()

    t_sum = operations[operation](t_a, t_b).tt_round(eps)
    t_sum_full = t_sum.full_tensor()

    print '\ntensor A ' + sign_map[operation] + ' B after truncation:'
    print t_sum.printable_basic_info()
    print "difference in frob. norm from it's original form:", \
        frobenius_norm(t_sum_full - operations[operation](A, B))