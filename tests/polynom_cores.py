import numpy as np
import mpmath as mp

from numpy import reshape

from tensor_train import TensorTrain, frobenius_norm

def sym_sum_poly(d, coeffs, bounds=[0, 1], discretization=10):
    p = coeffs.size - 1
    x = np.linspace(bounds[0], bounds[1], discretization, endpoint=False)

    # Make first core
    first = np.array([[np.sum([coeffs[k] * float(mp.binomial(k, s)) * x[i]**(k-s) for k in xrange(s, p+1)]) for s in xrange(p+1)] for i in xrange(discretization)])
    first = first[np.newaxis, ...]


    # Make binomial matrix B(i, j) = C_{i}^{i-j}
    #binomials = np.array([[np.float(mp.binomial(i, i-j)) for j in xrange(p+1)] for i in xrange(p+1)])
    xx = np.repeat(x, p+1)
    jj = np.tile(np.arange(p+1), discretization)
    bins = np.array([[np.float(mp.binomial(i, i-j)) for j in jj] for i in xrange(p+1)])
    mid_core = np.array([bins[i]*xx**np.clip((i-jj), a_min=0, a_max=p+1) for i in xrange(p+1)]).reshape((p+1, discretization, p+1))
    mid_core = reshape(mid_core, (p+1, discretization, p+1))

    # Make last core
    last = np.array([[xx**i for xx in x] for i in xrange(p+1)])[..., np.newaxis]
    return TensorTrain.from_cores([first] + [mid_core.copy() for _ in xrange(d-2)] + [last])



def verify_simple_poly_tensor(d, coeffs, discretization=10):
    tt = sym_sum_poly(d, coeffs, discretization=discretization)
    exact_representation = tt.full_tensor()
    d = tt.d
    n = tt.n
    p = coeffs.size - 1
    ethalon = np.zeros(tt.n)
    for i in np.ndindex(*n):
        multiplier = 1. / discretization
        arg = multiplier*np.sum(i)
        arg_powers = arg**(np.arange(p+1))
        ethalon[i] = np.sum(coeffs * arg_powers)
    nrm = np.linalg.norm(exact_representation - ethalon)
    print nrm
    return nrm < 1e-14