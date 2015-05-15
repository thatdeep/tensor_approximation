import numpy as np
import seaborn as sns
import cProfile

from function_approximation import InnerFunctionMp as InnerFunction
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

func = InnerFunction(N=2)

X = np.linspace(0.0, 1.0, 500, endpoint=False)
cProfile.run("np.array([float(func.evaluate(x)) for x in X])")
#Y = np.array([float(func.evaluate(x)) for x in X])

#sns.plt.plot(X, Y, '.')
#sns.plt.show()

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