import numpy as np
from numpy import random as rnd
from tensor_train.utils import frobenius_norm
from tensor_train.core import TensorTrain


def simple_operation_test(operation='add', percent=0.95, ndims=4, avg_dimsize=5, variate_dims=False, eps=1e-16):

    # TODO IT IS SHIT
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
        shape_form = tuple(avg_dimsize for _ in xrange(ndims))
    else:
        shape_form = tuple(rnd.randint(avg_dimsize / 2 + 1, avg_dimsize + avg_dimsize / 2, ndims))
    A = rnd.random(shape_form)
    B = rnd.random(shape_form)
    zero_idx = np.array(rnd.binomial(1, percent, shape_form), dtype=np.bool)
    A[zero_idx] = 0
    zero_idx = np.array(rnd.binomial(1, percent, shape_form), dtype=np.bool)
    B[zero_idx] = 0

    t_a = TensorTrain(A, eps)
    t_b = TensorTrain(B, eps)

    t_a_full = t_a.full_tensor()
    t_b_full = t_b.full_tensor()

    print '\ntensor A:'
    print t_a.printable_basic_info()
    print "difference in frob. norm from it's original form:", frobenius_norm(t_a_full - A)

    print '\ntensor B:'
    print t_b.printable_basic_info()
    print "difference in frob. norm from it's original form:", frobenius_norm(t_b_full - B)

    t_sum = operations[operation](t_a, t_b)
    t_sum_full = t_sum.full_tensor()

    print '\ntensor A ' + sign_map[operation] + ' B:'
    print t_sum.printable_basic_info()
    print "difference in frob. norm from it's original form:", \
        frobenius_norm(t_sum_full - operations[operation](A, B))

    t_sum_trunc = t_sum.tt_round(eps)
    t_sum_trunc_full = t_sum_trunc.full_tensor()

    print '\ntensor A ' + sign_map[operation] + ' B after truncation:'
    print t_sum_trunc.printable_basic_info()
    print "difference in frob. norm from it's original form:", \
        frobenius_norm(t_sum_trunc_full - operations[operation](A, B))

