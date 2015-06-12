import math
import numpy as np


# Run operation test
"""
from tests import operations_test

for operation_image in ['add', 'sub', 'mul']:
    simple_operation_test(operation_image)
"""


# Function approximation part
"""
from tests import test_sinus

test_sinus()
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
#"""
from tensor_train import TensorTrain, skeleton

A = np.random.random((30, 20, 40, 30))
t = TensorTrain(A)
skelet = skeleton(A)
#"""


# Small test of indexRC class
"""
from tensor_train import IndexRC

n = A.shape
ranks = np.array([7, 10,  8], dtype=int)
irc = IndexRC(n, ranks)
print irc.index
"""
