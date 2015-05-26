import math
import numpy as np
import mpmath as mp
from mpmath import workdps, mpf
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import cProfile

from function_approximation import Combinator
from function_approximation import InnerFunctionMp as InnerFunction, InnerFunctionG
#from tests import uniformly_distributed_approximation_test, small_decomposition_test, simple_operation_test

def ethalon_function(x, C=3):
    return (math.exp(-(C*x[0])**2)*(C*x[0])**2 + math.exp(-(C*x[1])**2)*(C*x[1])**2)

norm = 2.0 / math.e
L = 0.6

combinator = Combinator(ethalon_function, norm, L, N=2)

M = 2
x = np.linspace(0.0, 1.0, M, endpoint=False)
y = np.linspace(0.0, 1.0, M, endpoint=False)

X, Y = np.meshgrid(x, y)

points_raw = np.vstack([X.ravel(), Y.ravel()]).T
print points_raw.shape
values_raw = np.apply_along_axis(ethalon_function, 1, points_raw)
cProfile.run('[combinator(point) for point in points_raw]')
comb_values_raw = [combinator(point) for point in points_raw]
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

CS = plt.contourf(X, Y, V, cmap=plt.cm.bone, origin='lower')
# Make a colorbar for the ContourSet returned by the contourf call.
cbar = plt.colorbar(CS)
cbar.ax.set_ylabel('value')
plt.show()"""


"""func = InnerFunction(N=4)

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