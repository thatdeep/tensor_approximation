import math
import numpy as np


# Run operation test
"""
from tests import operations_test

for operation_image in ['add', 'sub', 'mul']:
    simple_operation_test(operation_image)
"""

"""
from tensor_train import frobenius_norm
from tests.sinus_cores import sym_sum_sinus_tensor
t1 = sym_sum_sinus_tensor(4)
t2 = sym_sum_sinus_tensor(4)
T = t1 - t2
print frobenius_norm(T.full_tensor())
print frobenius_norm(T)

tr = T.tt_round()
print frobenius_norm(tr), frobenius_norm(tr.full_tensor())
"""

#from tests import simple_arithmetic_test
#simple_arithmetic_test(d=4, n=10, operation='mul', percent=0.2, eps=1e-9)


# Function approximation part

"""
from tests import test_sinus
from multivariate_test import test_sinus_write

test_sinus_write(M=10)
"""


# Maxvol test part
"""
from tensor_train import maxvol

A = np.random.random((10, 4))
A = np.array([
    [0.602657, 0.287517, 0.179929, 0.060979],
    [0.508685, 0.163114, 0.676342, 0.093480],
    [0.288947, 0.668028, 0.055647, 0.181749],
    [0.141189, 0.251369, 0.682098, 0.323910],
    [0.947997, 0.588853, 0.308936, 0.589636],
    [0.243959, 0.893497, 0.613157, 0.225322],
    [0.934003, 0.301660, 0.725291, 0.536643],
    [0.934360, 0.295389, 0.485394, 0.829019],
    [0.631194, 0.492146, 0.607109, 0.388354],
    [0.808554, 0.767534, 0.315304, 0.391934]])
dets, d = maxvol_test(A)
print dets
print d
"""


# Verifying analytical exact sinus representation
"""
from tests import verify_simple_sinus_tensor

print [verify_simple_sinus_tensor(d, 10) for d in xrange(2, 5)]
"""


# Testing skeleton decomposition
"""
from tensor_train import TensorTrain, skeleton, frobenius_norm
from tests.sinus_cores import sym_sum_sinus_tensor

#A = np.random.random((30, 20, 40, 30))
#t = TensorTrain(A)
t = sym_sum_sinus_tensor(8)
skelet = skeleton(t.full_tensor())
xx = (skelet - t).full_tensor()
print frobenius_norm((skelet - t).tt_round(eps=1e-12))
print np.max(np.abs(xx))
print frobenius_norm(skelet.full_tensor() - t.full_tensor())
"""


from tensor_train import TensorTrain, frobenius_norm
from tensor_train import BlackBox
from tests.sinus_cores import sym_sum_sinus_tensor

def f(x):
    return np.sin(np.sum(x))

def f_vect(x):
    arg = np.sum(x, axis=0)
    return np.sin(arg)

def test_f_sym(f, f_vect, d, bounds=None, discr=10, eps=1e-9):
    if bounds is None:
        bounds = [0, 1]
    #n = tuple([discr]*d)
    space = np.linspace(bounds[0], bounds[1], discr, endpoint=False)
    black_box = BlackBox(f, f_vect, bounds, discr, d, dtype=np.float, array_based=False)
    return TensorTrain(black_box, eps=eps)

d = 305
discr  = 10
eps = 1e-7
t_exact = sym_sum_sinus_tensor(d, discretization=discr)
t_approx = test_f_sym(f, f_vect, d, discr=discr, eps=eps)
print t_approx.r

#t_approx = t_approx.tt_round()
print frobenius_norm(t_exact)
print frobenius_norm(t_approx)
print frobenius_norm(t_exact - t_approx), eps*frobenius_norm(t_exact)


"""
def f(x):
    return np.sum(x)**5

dims = tuple([10, 20, 10, 10])
d = len(dims)

bounds = np.vstack([[0, 5], [0, 10], [0, 5], [0, 5]])
spaces = [np.linspace(bound[0], bound[1], dim, endpoint=False) for (bound, dim) in zip(bounds, dims)]
def f_box_generator(f, spaces):
    return lambda i: f([spaces[idx][ii] for idx, ii in enumerate(i)])

f_box = f_box_generator(f, spaces)

black_box = BlackBox(f_box, dims, d, dtype=np.float, array_based=False)
t = TensorTrain(black_box)

F = np.fromiter((f([spaces[idx][ii] for idx, ii in enumerate(i)]) for i in np.ndindex(*dims)), dtype=np.float).reshape(dims)
print F.shape
tr = t.tt_round(eps=1e-9)
print tr.r
TF = tr.full_tensor()
D = TF - F
print frobenius_norm(D)
print np.max(np.abs(D))
"""

# Small test of indexRC class
"""
from tensor_train import TensorTrain, frobenius_norm
from tensor_train import BlackBox

def f(x):
    arg = np.sum(x)
    if arg == 0: arg = 1
    return np.sin(1.0 / arg)

def f_vect(x):
    arg = np.sum(x, axis=0)
    arg[arg == 0] = 1
    return np.sin(1.0 / arg)


def test_f_sym(f, f_vect, d, bounds=None, discr=10, eps=1e-9):
    if bounds == None:
        bounds = [0, 1]
    space = np.linspace(bounds[0], bounds[1], discr, endpoint=False)
    black_box = BlackBox(f, f_vect, bounds, discr, d, dtype=np.float, array_based=False)
    return TensorTrain(black_box, eps=eps), black_box

xxx = []

import cProfile

for d in [10, 20, 50]:
    dims = tuple([10]*d)
    discr  = 10
    eps = 1e-5

    #t_exact = sym_sum_sinus_tensor(d, discretization=discr)
    #cProfile.run('t_approx, f_exact = test_f_sym(f, f_vect, d, discr=discr, eps=eps)')
    t_approx, f_exact = test_f_sym(f, f_vect, d, discr=discr, eps=eps)
    print t_approx.r

    maxx = 0
    space = np.linspace(0, 1, discr, endpoint=False)
    sample_size = 10000
    random_samples = np.vstack([np.random.randint(0, dims[i], sample_size) for i in xrange(len(dims))]).T
    for sample in random_samples:
        #f_sample = tuple([space[i] for i in sample])
        xx =  np.abs(f_exact[sample] - t_approx[sample])
        if xx > maxx:
            maxx = xx
    xxx.append(xx)
    print xx

print xxx
"""
#t_approx = t_approx.tt_round()
#print frobenius_norm(t_exact)
#print frobenius_norm(t_approx)
#print frobenius_norm(t_exact - t_approx), eps*frobenius_norm(t_exact)

"""
from tensor_train import TensorTrain, frobenius_norm
from tensor_train import BlackBox

def f(x):
    arg = np.sum(x)
    if arg == 0: arg = 1
    return np.sin(1.0 / arg)

dims = tuple([10, 20, 10, 10, 10, 10])
d = len(dims)

bounds = np.vstack([[0, 5], [0, 10], [0, 5], [0, 5], [0, 5], [0, 5]])
spaces = [np.linspace(-bound[1], bound[0], dim, endpoint=False) for (bound, dim) in zip(bounds, dims)]
def f_box_generator(f, spaces):
    return lambda i: f([spaces[idx][ii] for idx, ii in enumerate(i)])

f_box = f_box_generator(f, spaces)

eps = 1e-7
black_box = BlackBox(f_box, dims, d, dtype=np.float, array_based=False)
t = TensorTrain(black_box, eps=eps)

F = np.fromfile('../f.arr').reshape(dims)
#F = np.fromiter((f([spaces[idx][ii] for idx, ii in enumerate(i)]) for i in np.ndindex(*dims)), dtype=np.float).reshape(dims)
print F.shape
print t.r
print frobenius_norm(F)
print frobenius_norm(t)
T = t.full_tensor()
D = T - F
print frobenius_norm(D)
print eps * frobenius_norm(F)
print np.max(np.abs(D))

print '-'*80

tr = t.tt_round(eps=eps)
print tr.r
TF = tr.full_tensor()
D = TF - F
print frobenius_norm(D)
print np.max(np.abs(D))

idx = [0]*6
print F[idx]
print t[idx]
"""

"""
from tensor_train import IndexRC

n = A.shape
ranks = np.array([7, 10,  8], dtype=int)
irc = IndexRC(n, ranks)
print irc.index
"""
