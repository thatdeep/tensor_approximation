import numpy as np

from numpy import float128 as f128
from utils import x_decomposition, x_restore, power_2_bound
import decimal as dc


def exact_b_list(gamma_orig, N, max_k=8, prec=1024):
    dc.getcontext().prec=prec
    gamma = dc.Decimal(gamma_orig)
    betas = [dc.Decimal(0), dc.Decimal(1)]
    C1 = dc.Decimal(2)
    for k in xrange(2, max_k, 1):
        new_bk = betas[-1] * N + C1 + 1
        #if new_bk > 1024:
        #    return np.array(betas)
        #if not gamma ** (-new_bk) < gamma**(-N * betas[-1] - C1):
            #print gamma ** (-new_bk)
            #print gamma**(-N * betas[-1] - C1)
        assert gamma ** (-new_bk) < gamma**(-N * betas[-1] - C1)
        print new_bk
        betas.append(new_bk)
    #print betas
    return betas


def exact_generate_ps(blist, bits_orig, gamma_int=2, prec=1024):
    ## Just expand bits array with zero, if it will bring errors
    kron = lambda x, y: 1 if x == y else 0
    bits = bits_orig
    if len(blist) > len(bits):
        bits = list(bits_orig) + [0]*(len(blist) - len(bits))
    dc.getcontext().prec=prec
    gamma = dc.Decimal(gamma_int)
    max_k = len(blist)
    ms = [0]*max_k
    ps = [0]*max_k
    ms[0] = dc.Decimal(gamma)
    ps[0] = dc.Decimal(1)
    for k in xrange(1, max_k):
        ms[k] = (gamma**(blist[k] - blist[k-1]))*(dc.Decimal(R(k-1, k-1, bits, gamma) * (ms[k-1] - gamma))/2 + 1)
        ps[k] = (kron(bits[k], gamma - 1))*((gamma - 2 + ms[k-1])) / 2 + bits[k] * (1 - kron(bits[k], gamma - 1))
    return ps


def exact_inner_function(x, N=2, gamma=None):
    if gamma == None:
        gamma = power_2_bound(2*N + 2)
    x_sign, x_bits = x_decomposition(x, gamma)
    exact_blist = exact_b_list(gamma, 2*N + 2)
    exact_ps = exact_generate_ps(exact_blist, x_bits, gamma_int=gamma)
    powers = [dc.Decimal(gamma)]*len(exact_blist)
    #print len(exact_blist)
    return (-1)**x_sign * sum(map(lambda (c, power_mask, power): c * power_mask ** (-power), zip(exact_ps, powers, exact_blist)))
    #return map(lambda (c, power_mask, power): c * power_mask ** (-power), zip(exact_ps, powers, exact_blist))


def simple_b_list(gamma_orig, N, max_k=20):
    gamma = f128(gamma_orig)
    betas = [f128(0), f128(1)]
    for k in xrange(2, max_k, 1):
        C1 = 2
        new_bk = betas[-1] * N + C1 + 1
        #print new_bk
        if new_bk > 1000:
            return np.array(betas)
        assert gamma**(-new_bk) < gamma**(-N * betas[-1] - C1)
        betas.append(new_bk)
    return np.array(betas, dtype=f128)


def R(n, j, i_m, gamma=2):
    kron = lambda x, y: 1 if x == y else 0
    return reduce(lambda x, y: x*y, [kron(i_m[m], gamma - 1) + kron(i_m[m], gamma - 2) for m in xrange(n, j + 1, 1)])


def generate_ps(blist, bits, gamma=2):
    kron = lambda x, y: 1 if x == y else 0
    max_k = len(blist)
    ms = np.zeros(max_k)
    ps = np.zeros(max_k)
    ms[0] = gamma
    ps[0] = 1
    for k in xrange(1, max_k):
        ms[k] = 1.0*(gamma**(blist[k] - blist[k-1]))*(1.0/2 * R(k-1, k-1, bits, gamma) * (ms[k-1] - gamma) + 1)
        ps[k] = 1.0*(kron(bits[k], gamma - 1))*(1.0 / 2 * (gamma - 2 + ms[k-1])) + bits[k] * (1 - kron(bits[k], gamma - 1))
    return ps

def inner_function(x, N=2, gamma=None):
    if gamma == None:
        gamma = power_2_bound(2*N + 2)
    x_sign, x_bits = x_decomposition(x, gamma)
    blist = simple_b_list(gamma, 2*N + 2)
    print blist, gamma
    ps = generate_ps(blist, x_bits)
    powers = np.zeros(len(blist), dtype=np.float128)
    powers[:] = gamma
    return (-1)**x_sign * (np.sum(ps * powers ** (-blist)))