import math
import numpy as np

from function_approximation import Composer


def test_sinus():
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