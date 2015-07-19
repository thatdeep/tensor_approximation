import math
import numpy as np

import mpmath as mp
from mpmath import workdps, mpf
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from function_approximation import InnerFunctionMp as InnerFunction, InnerFunctionG



func = InnerFunction(N=2)
sns.set_style("ticks", {"xtick.major.size": 16, "ytick.major.size": 16})
rc={'axes.labelsize': 20, 'font.size': 24, 'legend.fontsize': 24.0, 'axes.titlesize': 24}
sns.set(rc=rc)

with workdps(2048):
    #point = mpf('0.59349056')
    #eps = mpf('2.0')**(-10)
    #X = mp.linspace(mpf('0.5934905'), mpf('0.5934907'), 2000, endpoint=False)
    X = mp.linspace(0, 1, 5000, endpoint=False)
    #cProfile.run("np.array([float(func.evaluate(x)) for x in X])")
    Y = np.array([float(func(x)) for x in X])

    #print func.list_of_betas(dps=40000)

    sns.plt.plot(X, Y)
    sns.plt.axes().set_ylabel(r'$\Psi(x)$')
    sns.plt.axes().set_xlabel(r'$x$')
    sns.plt.savefig('inner_2d.png', dps=300, transparent=True)
    sns.plt.show()

"""
N = 2

discr = 1000
func = InnerFunctionG(N=2)

x = np.linspace(0.0, 1.0, 50, endpoint=False)
y = np.linspace(0.0, 1.0, 50, endpoint=False)

X ,Y  = np.meshgrid(x, y)

points = np.vstack([X.ravel(), Y.ravel()]).T

values = np.apply_along_axis(lambda p: float(func(p, 2)), 1, points)

V = values.reshape(X.shape)

fig = plt.figure()

#---- First subplot
ax = fig.add_subplot(1, 1, 1, projection='3d')
surf = ax.plot_surface(X, Y, V, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
ax.set_zlim3d(V.min(), V.max())
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
plt.savefig('inner_func_g.png', dps=300, transparent=True)
plt.show()
"""

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

print [verify_simple_sinus_tensor(d, discretization=10) for d in xrange(2, 5)]
"""


"""
from tests import verify_simple_poly_tensor

coeffs = np.array([0, 0, 1])

print [verify_simple_poly_tensor(d, coeffs, discretization=10) for d in xrange(2, 6)]
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

"""
from tensor_train import TensorTrain, frobenius_norm
from tensor_train import BlackBox
from tests.polynom_cores import sym_sum_poly

def f_gen(coeffs):
    p = coeffs.size - 1
    def f(x):
        arg = np.sum(x)
        return np.sum(coeffs * arg**(np.arange(p+1)))
    return f

def f_vect_gen(coeffs):
    p = coeffs.size - 1
    def f(x):
        # arg will have size of m
        arg = np.sum(x, axis=0)
        val_space = np.vstack([coeffs[i]*arg**i for i in xrange(p+1)])
        return np.sum(val_space, axis=0)
    return f

def test_f_sym(f, f_vect, d, bounds=None, discr=10, eps=1e-9):
    if bounds is None:
        bounds = [0, 1]
    #n = tuple([discr]*d)
    space = np.linspace(bounds[0], bounds[1], discr, endpoint=False)
    black_box = BlackBox(f, f_vect, bounds, discr, d, dtype=np.float, array_based=False)
    return TensorTrain(black_box, eps=eps)


dd = range(10, 110, 10)
pp = range(1, 9, 1)

form = (len(pp), len(dd))

norm_exact = np.zeros(form)
norm_approx = np.zeros(form)
deltas = np.zeros(form)
eps_norm = np.zeros(form)

discr = 20
bounds=[-1, 1]
eps=1e-5
for i, p in enumerate(pp):
    coeffs = np.ones(p+1, dtype=int)
    #coeffs[1::] = -1
    for j, d in enumerate(dd):
        if i == 7 and j == 8:
            continue
        print "I:{i}, J:{j}".format(i=i, j=j)
        t_exact = sym_sum_poly(d, coeffs, bounds=bounds, discretization=discr)
        t_approx = test_f_sym(f_gen(coeffs), f_vect_gen(coeffs), d, bounds=bounds, discr=discr, eps=eps)
        norm_exact[i, j] = frobenius_norm(t_exact)
        norm_approx[i, j] = frobenius_norm(t_approx)
        deltas[i, j] = frobenius_norm(t_exact - t_approx)
        eps_norm[i, j] = eps*norm_exact[i, j]
        print t_exact.r
        print t_approx.r
        print '-'*80

norm_exact.tofile('./poly_tests_data/i_norm_exact_{n}'.format(n=discr))
norm_approx.tofile('./poly_tests_data/i_norm_approx_{n}'.format(n=discr))
deltas.tofile('./poly_tests_data/i_deltas_{n}'.format(n=discr))
eps_norm.tofile('./poly_tests_data/i_eps_norm_{n}'.format(n=discr))

p = pp[7]
d = dd[8]
coeffs = np.ones(p+1, dtype=int)
t_exact = sym_sum_poly(d, coeffs, bounds=bounds, discretization=discr)
t_approx = test_f_sym(f_gen(coeffs), f_vect_gen(coeffs), d, bounds=bounds, discr=discr, eps=eps)

nexact = frobenius_norm(t_exact)
print nexact
print frobenius_norm(t_approx)
print frobenius_norm(t_exact - t_approx)
print eps*nexact
"""


"""
d = 100
discr  = 20
p = 10
coeffs = np.ones(p+1, dtype=int)
#coeffs[-1] = 1
eps = 1e-7
t_exact = sym_sum_poly(d, coeffs, bounds=[0, 10], discretization=discr)
t_approx = test_f_sym(f_gen(coeffs), f_vect_gen(coeffs), d, bounds=[0, 10], discr=discr, eps=eps)
print t_approx.r

#t_approx = t_approx.tt_round()
print frobenius_norm(t_exact)
print frobenius_norm(t_approx)
print frobenius_norm(t_exact - t_approx), eps*frobenius_norm(t_exact)

ttt = t_exact.tt_round(eps=eps)
print frobenius_norm(t_exact - ttt), eps*frobenius_norm(t_exact)
"""

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
        bounds = [-1, 0]
    space = np.linspace(bounds[0], bounds[1], discr, endpoint=False)
    black_box = BlackBox(f, f_vect, bounds, discr, d, dtype=np.float, array_based=False)
    return TensorTrain(black_box, eps=eps), black_box

xxx = []
norms = []

import cProfile

for d in [10, 100, 2000][-1:]:
    discr  = 10
    dims = tuple([discr]*d)
    eps = 1e-9

    #t_exact = sym_sum_sinus_tensor(d, discretization=discr)
    cProfile.run('t_approx, f_exact = test_f_sym(f, f_vect, d, discr=discr, eps=eps)')
    t_approx, f_exact = test_f_sym(f, f_vect, d, discr=discr, eps=eps)
    print t_approx.r

    maxx = 0
    space = np.linspace(-1, 0, discr, endpoint=False)
    sample_size = 10000
    random_samples = np.vstack([np.random.randint(0, dims[i], sample_size) for i in xrange(len(dims))]).T
    for sample in random_samples:
        #f_sample = tuple([space[i] for i in sample])
        fe = f_exact[sample]
        xx =  np.abs((fe - t_approx[sample]) / fe)
        if xx > maxx:
            maxx = xx
    xxx.append(xx)
    norms.append(frobenius_norm(t_approx))
    print xx

print xxx
print norms
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
