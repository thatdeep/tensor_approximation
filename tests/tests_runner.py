import numpy as np
import seaborn as sns

from function_approximation import inner_function
from tests import uniformly_distributed_approximation_test, small_decomposition_test, simple_operation_test

# Run operation test

for operation_image in ['add', 'sub', 'mul']:
    simple_operation_test(operation_image)


# Run approximation test

# size of a uniformly distributed set of doubles
test_size = 500
# high and low bounds are [-bound, +bound]
bound = 10000

uniformly_distributed_approximation_test(size=test_size, bound=bound)
small_decomposition_test()

# Function approximation part

x = np.random.random() * 100
print x

arr = np.array([inner_function(x) for i in range(1000)])
sns.distplot(arr)