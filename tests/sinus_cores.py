import numpy as np
import mpmath as mp

from mpmath import workdps, mpf

from tensor_train import TensorTrain, frobenius_norm

"""
def sum_sinus_tensor(requested_shape, bounds=[0, 1], discretization=10):
    n = tuple(requested_shape)
    d = len(n)
    bounds = np.asarray(bounds)
    if len(bounds.shape) == 1:
        bounds = np.tile(bounds).reshape((d, -1))
    assert bounds.shape == (d, 2),\
        "given bounds must have {d} dimensions, but have only {dd}".format(d=d, dd=bounds.shape[0])

    discretization = np.asarray(discretization)
    if discretization.size == 1:
        discretization = [discretization] * d
    assert discretization.shape == (d,),\
        "discretization must be a positive int number, or array of d numbers"

    spaces = [np.linspace(bound[0], bound[1]) for bound in bounds]

    sxs = []
"""

def sym_sum_sinus_tensor(d, bounds=[0, 1], discretization=10):
    space = np.linspace(bounds[0], bounds[1], discretization, endpoint=False)
    sinx, cosx = np.sin(space), np.cos(space)

    first_core = np.vstack([sinx, cosx]).T[np.newaxis, ...]
    last_core = np.vstack([cosx, sinx])[..., np.newaxis]
    middle_core = np.hstack([np.vstack([cosx, sinx]),
                             np.vstack([-sinx, cosx])])
    middle_core = np.reshape(middle_core, (2, 2, -1)).transpose((0, 2, 1))

    return TensorTrain.from_cores([first_core] + [middle_core.copy() for _ in xrange(d - 2)] + [last_core])

def sym_sum_sinus_normed_tensor(d, bounds=[0, 1], discretization=10):
    dps=500
    space = np.linspace(bounds[0], bounds[1], discretization, endpoint=False)
    mult = np.sqrt(discretization)
    mult2 = np.sqrt(2*discretization)
    sinx, cosx = np.sin(space), np.cos(space)

    first_core = np.vstack([sinx, cosx]).T[np.newaxis, ...] / mult
    last_core = np.vstack([cosx, sinx])[..., np.newaxis] / mult
    middle_core = np.hstack([np.vstack([cosx, sinx]),
                             np.vstack([-sinx, cosx])]) / (mult2)
    middle_core = np.reshape(middle_core, (2, 2, -1)).transpose((0, 2, 1))

    with workdps(dps):
        norm = mpf(mult2)
        norm = norm**(d-2)
        norm *= mult**2

    return TensorTrain.from_cores([first_core] + [middle_core.copy() for _ in xrange(d - 2)] + [last_core], norm)


def verify_simple_sinus_tensor(d, discretization=10):
    tt = sym_sum_sinus_tensor(d, discretization=discretization)
    exact_representation = tt.full_tensor()
    d = tt.d
    n = tt.n
    ethalon = np.zeros(tt.n)
    for i in np.ndindex(*n):
        multiplier = 1. / discretization
        ethalon[i] = np.sin(multiplier*np.sum(i))
    nrm = np.linalg.norm(exact_representation - ethalon)
    print nrm
    return nrm < 1e-14


def yyy(d, discretization=10):
    shp = tuple([discretization]*d)
    ethalon = np.zeros(shp)
    for i in np.ndindex(*shp):
        multiplier = 1. / discretization
        ethalon[i] = np.sin(multiplier*np.sum(i))
    return np.linalg.norm(ethalon)

def yy(d, discretization=10):
    shp = tuple([discretization]*d)
    ethalon = np.zeros(shp)
    for i in np.ndindex(*shp):
        multiplier = 1. / discretization
        ethalon[i] = np.sin(multiplier*np.sum(i))
    return ethalon