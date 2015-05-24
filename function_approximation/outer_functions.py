import math
import numpy as np
import mpmath as mp

from mpmath import mpf, workdps




def cnst(gamma, k):
    print 'gamma: {gamma}, k: {k}'.format(gamma=gamma, k=k)
    return 2**k * (mpf('0.5')*(gamma - 3) / (gamma - 1) * gamma**(-k))**(mp.ln(2) / mp.ln(gamma))

"""def dot_sequence(indices, gamma, eps, k_max=15):
    N = indices.size
    q_size = 2*N
    values = [dc.Decimal(idx) + dc.Decimal(1) / dc.Decimal(gamma - 1) for idx in indices]
    dots = [[[dc.Decimal(0) for i in xrange(N)] for k in k_max] for q in q_size]

    for q in xrange(N):
        eq = dc.Decimal(eps)*dc.Decimal(q)
        for k in xrange(k_max):
            for i, idx in enumerate(indices):
                dots[q][k].append(values[i] - eq)
            values[i] /= dc.Decimal(gamma)
    return dots"""


class Outer(object):
    def __init__(self, f, k, N=2, dps=2048):
        self.f = f
        self.k = k
        self.dps=dps

    def __call__(self, x):
        pass

    def index_search(self, x, gamma, k, N=2):
        with workdps(self.dps):
            def q_estimate_from_many(ns):
                mins = [-1]*len(ns)
                maxs = [2*N + 1]*len(ns)
                for i, n in enumerate(ns):
                    if n == gamma - 2:
                        mins[i] = 0
                        maxs[i] = 0
                        continue
                    if gamma - 3 - n >= 0:
                        mins[i] = gamma - 3 - n
                    if gamma - 2 - n < 2*N:
                        maxs[i] = gamma - 1 - n

                print [el >= 0 for el in mins]
                if all([el >= 0 for el in mins]):
                    return 0

                if all([el < 2*N + 1 for el in maxs]):
                    return max(maxs)
                print ns
                print mins
                print maxs
                raise Exception('wrong way of counting q')

            x_big = [mpf(el) * gamma**k for el in x]
            eps = mpf('1') / (gamma - 1)
            intervals = mp.linspace(mpf('0'), mpf('1'), gamma)

            x_big_frac = [mp.frac(el) for el in x_big]
            numbers_of_intervals = [mp.floor(el / eps) for el in x_big_frac]

            q = q_estimate_from_many(numbers_of_intervals)

            return [int(mp.floor(el - eps*q*gamma**k)) for el in x_big], q