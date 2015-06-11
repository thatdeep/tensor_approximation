import math
import numpy as np

from tensor_train import maxvol
from tests import maxvol_test
import mpmath as mp
from mpmath import workdps, mpf
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import cProfile

from tests import approx_test
from function_approximation import Combinator, Composer
from function_approximation import InnerFunctionMp as InnerFunction, InnerFunctionG
#from tests import uniformly_distributed_approximation_test, small_decomposition_test, simple_operation_test

#def ethalon_function(x, C=3):
#    return (math.exp(-(C*x[0])**2)*(C*x[0])**2 + math.exp(-(C*x[1])**2)*(C*x[1])**2)

#norm = 2.0 / math.e
#L = 0.6

"""
def ethalon_function(x):
    return math.sin(x[0] + x[1])/2

norm = 0.5
L = math.sqrt(2.0)/2

combinator = Composer(ethalon_function, norm, L, N=2, initial_k=10)

M = 10
x = np.linspace(0.0, 1.0, M, endpoint=False)
y = np.linspace(0.0, 1.0, M, endpoint=False)

X, Y = np.meshgrid(x, y)

points_raw = np.vstack([X.ravel(), Y.ravel()]).T
values_raw = np.apply_along_axis(ethalon_function, 1, points_raw)
V = values_raw.reshape(X.shape)

for k in xrange(1, 10):
    combinator = Composer(ethalon_function, norm, L, N=2, initial_k=k)
    comb_values_raw = [combinator(point) for point in points_raw]
    CV = np.array(comb_values_raw, dtype=np.float).reshape(X.shape)
    print ('{k}\t{max_err}\t{err_norm}\n'+'*'*80).format(k=k, max_err=np.max(np.abs(V - CV)), err_norm=np.linalg.norm(V - CV))
"""

"""
def ethalon_function(x):
    return math.sin(x[0] + x[1])/2

norm = 0.5
L = math.sqrt(2.0)/2

combinator = Composer(ethalon_function, norm, L, N=2, initial_k=1)

M = 10
x = np.linspace(0.0, 1.0, M, endpoint=False)
y = np.linspace(0.0, 1.0, M, endpoint=False)

X, Y = np.meshgrid(x, y)

#cProfile.run('combinator([0.5, 0.5])')

points_raw = np.vstack([X.ravel(), Y.ravel()]).T
print points_raw.shape
values_raw = np.apply_along_axis(ethalon_function, 1, points_raw)
#cProfile.run('[combinator(point) for point in points_raw]')
#comb_values_raw = [combinator(point) for point in points_raw]
cvrs = []
for kk in xrange(1, 4):
    combinator = Composer(ethalon_function, norm, L, N=2, initial_k=1, amount_of_k=kk)
    cvrs.append([combinator(point) for point in points_raw])
cvrss = np.array(cvrs, dtype=np.float)
comb_values_raw = np.average(cvrss, axis=0)
print values_raw
print comb_values_raw
V = values_raw.reshape(X.shape)
CV = np.array(comb_values_raw, dtype=np.float).reshape(X.shape)

print np.linalg.norm(CV - V)
print np.max(np.abs(CV - V))

# Twice as wide as it is tall.
fig = plt.figure(figsize=plt.figaspect(0.5))

#---- First subplot
ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(X, Y, V, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
ax.set_zlim3d(V.min(), V.max())

fig.colorbar(surf, shrink=0.5, aspect=10)

#---- Second subplot
ax = fig.add_subplot(1, 2, 2, projection='3d')

surf = ax.plot_surface(X, Y, CV, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=10)

ax.set_zlim3d(CV.min(), CV.max())

plt.show()
"""

#CS = plt.contourf(X, Y, V, cmap=plt.cm.bone, origin='lower')
# Make a colorbar for the ContourSet returned by the contourf call.
#cbar = plt.colorbar(CS)
#cbar.ax.set_ylabel('value')

# Run operation test
"""
for operation_image in ['add', 'sub', 'mul']:
    simple_operation_test(operation_image)
"""
"""
# Run approximation test

# size of a uniformly distributed set of doubles
test_size = 500
# high and low bounds are [-bound, +bound]
bound = 10000

uniformly_distributed_approximation_test(size=test_size, bound=bound)
small_decomposition_test()
"""
# Function approximation part

"""func = InnerFunctionG(N=2)

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
plt.show()"""

#CS = plt.contourf(X, Y, V, cmap=plt.cm.bone, origin='lower')
# Make a colorbar for the ContourSet returned by the contourf call.
#cbar = plt.colorbar(CS)
#cbar.ax.set_ylabel('value')
#plt.show()

"""
func = InnerFunction(N=4)


print float(func(mpf('0.5898437499999999')))
print float(func(mpf('0.59')))"""



"""

print float(func(0.101))
print float(func(0.102))
print float(func(0.128))
print float(func(0.129))

with workdps(2048):
    point = mpf('0.59349056')
    eps = mpf('2.0')**(-10)
    #X = mp.linspace(mpf('0.5934905'), mpf('0.5934907'), 2000, endpoint=False)
    X = mp.linspace(point - eps, point + eps, 2000, endpoint=False)
    #cProfile.run("np.array([float(func.evaluate(x)) for x in X])")
    Y = np.array([float(func(x)) for x in X])
    print func(X[0]), func(X[-1]), func(point)

    #print func.list_of_betas(dps=40000)

    sns.plt.plot(X, Y, '.')
    sns.plt.show()"""

#x = np.linspace(0, 1.0, 5)
#y = np.array([exact_inner_function(i) for i in x], dtype=np.float64)

#sns.plt.plot(x, y, 'o')
#sns.plt.show()

#x = np.random.random() * 100
#ex = exact_inner_function(x)
#print x
#print ex

#arr = np.array([exact_inner_function(x) for x in np.random.random(100)], dtype=np.float64)
#sns.distplot(arr, bins=250)
#sns.plt.show()

"""
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

"""
A = np.random.random((10, 4))
print A
approx_test(A, 3)
"""


from tensor_train import TensorTrain, tt_qr, skeleton_decomposition, IndexRC

#A = np.random.random((30, 20, 40, 30))
#t = TensorTrain(A)
#skeleton_decomposition(A)

#n = A.shape
#ranks = np.array([7, 10,  8], dtype=int)
#irc = IndexRC(n, ranks)
#print irc.index

from tests import verify_simple_sinus_tensor

print [verify_simple_sinus_tensor(d, 10) for d in xrange(2, 7)]