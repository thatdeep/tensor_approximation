import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import cProfile

from function_approximation import InnerFunctionMp as InnerFunction, InnerFunctionG
#from tests import uniformly_distributed_approximation_test, small_decomposition_test, simple_operation_test

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

func = InnerFunctionG(N=2)

x = np.linspace(0.0, 1.0, 50, endpoint=False)
y = np.linspace(0.0, 1.0, 50, endpoint=False)

X ,Y  = np.meshgrid(x, y)

points = np.vstack([X.ravel(), Y.ravel()]).T

values = np.apply_along_axis(lambda p: float(func(p, 1)), 1, points)

V = values.reshape(X.shape)

CS = plt.contourf(X, Y, V, cmap=plt.cm.bone, origin='lower')
# Make a colorbar for the ContourSet returned by the contourf call.
cbar = plt.colorbar(CS)
cbar.ax.set_ylabel('value')
plt.show()


"""func = InnerFunction(N=2)

X = np.linspace(0.0, 1.0, 5000, endpoint=False)
#cProfile.run("np.array([float(func.evaluate(x)) for x in X])")
Y = np.array([float(func.evaluate(x)) for x in X])

#print func.list_of_betas(dps=40000)

sns.plt.plot(X, Y, '.')
sns.plt.show()
"""
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