import math
import numpy as np
import mpmath as mp

from mpmath import mpf, workdps
from utils import gamma_estimate
from itertools import combinations
from k_estimator import KseqEstimator

class CubeDispatcher(object):
    def __init__(self, k, N=2, dps=2048):
        with workdps(dps):
            self.N = N
            self.gamma = mpf(gamma_estimate(N))
            self.dps = dps
            self.k = k

    def index_search(self, x):
        with workdps(self.dps):
            def q_estimate_from_many(ns):
                qs_mask = np.ones(2 * self.N + 1, dtype=np.bool)
                for n in ns:
                    if n != self.gamma - 2 and n >= self.gamma - 2 - 2*self.N:
                        qs_mask[self.gamma - 2 - n] = False
                return qs_mask

            x_big = [mpf(el) * self.gamma**self.k for el in x]
            eps = mpf('1') / (self.gamma - 1)
            intervals = mp.linspace(mpf('0'), mpf('1'), self.gamma)

            x_big_frac = [mp.frac(el) for el in x_big]
            numbers_of_intervals = [mp.floor(el / eps) for el in x_big_frac]

            qs_mask = q_estimate_from_many(numbers_of_intervals)
            #print qs
            qs = np.array(range(2 * self.N + 1))[qs_mask]
            return [([int(mp.floor(el + eps * q * self.gamma**self.k)) for el in x_big], q) for q in qs], qs_mask

    def forbidden_index_search(self, x, q):
        with workdps(self.dps):
            x_big = [mpf(el) * self.gamma**self.k for el in x]
            eps = mpf('1') / (self.gamma - 1)
            intervals = mp.linspace(mpf('0'), mpf('1'), self.gamma)
            length = mpf('1.0') * (self.gamma - 2) / (self.gamma - 1)

            x_big_frac = [mp.frac(el) for el in x_big]
            ineq = [mp.frac(el + q*eps) <= length for el in x_big_frac]

            # Determine i-s
            ord_val = eps * q * self.gamma**self.k
            mod_val = ord_val - length

            indices = [int(mp.floor(el + ord_val)) if ineq[i] else int(mp.floor(el + mod_val)) for i, el in enumerate(x_big)]
            multipliers = [1]*len(x_big)

            for i in xrange(len(x_big)):
                if not ineq[i]:
                    indices[i] = [indices[i], indices[i] + 1]
                    multipliers[i] = [(indices[i][0] + eps - (x_big[i] + mod_val))/eps, (x_big[i] + mod_val - indices[i][0])/eps]

            # Now we have indices, where indices[i] may be number, or a pair of numbers (if x[i] lay between intervals)
            return indices, multipliers


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
        with workdps(dps):
            self.f = f
            self.k = k
            self.dps = dps
            self.N = N
            self.gamma = mpf(gamma_estimate(N))
            self.cube_dispatcher = CubeDispatcher(k, N, dps)
            self.mul_const = self.gamma**(-self.k)

    def __call__(self, x):
        with workdps(self.dps):
            indices, qs_mask = self.cube_dispatcher.index_search(x)
            q_forbidden = np.where(qs_mask == False)[0]
            sum_for_q = mpf('0.0')
            for i, q in indices:
                sum_for_q += self.exact_on_cube(i, q)
            for q in q_forbidden:
                idx, mlt = self.cube_dispatcher.forbidden_index_search(x, q)
                sum_for_q += self.average_between_cubes(idx, mlt, q)
            return sum_for_q

    def average_between_cubes(self, indices, multipliers, q):
        with workdps(self.dps):
            eps = mpf('1.0') / (self.gamma - 1)
            ind = [[i] if type(i) != list else i for i in indices]
            mlt = [[m] if type(m) != list else m for m in multipliers]
            shp = [len(i) for i in ind]
            f_value = mpf('0.0')
            #mul_const = self.gamma**(-self.k)
            m = np.prod(shp)
            for i in np.ndindex(*shp):
                point = [(ind[s][i[s]] + eps) * self.mul_const - eps*q for s in xrange(self.N)]
                #print point
                f_value += mpf(self.f(point)) * np.prod([mlt[s][i[s]] for s in xrange(self.N)]) / (self.N + 1)
            f_value /= m
            return f_value

    def exact_on_cube(self, index, q):
        with workdps(self.dps):
            eps = mpf('1.0') / (self.gamma - 1)
            #mul_const = self.gamma**(-self.k)
            point = [(index[s] + eps) * self.mul_const - eps * q for s in xrange(self.N)]
            return mpf(self.f(point)) / (self.N + 1)


"""class Combinator(object):
    def __init__(self, f_ethalon, f_norm, L, N=2, dps=2048*4):
        self.f = f_ethalon
        self.N = N
        self.f_norm = f_norm
        self.L = L
        kseq_estimator = KseqEstimator(f_norm, L, N, dps)
        kseq, norms = kseq_estimator.estimate_kseq(amount_of_k=3)
        def generate_inverted(f):
            return lambda x: x - f(x)
        current_func = f_ethalon
        for k in kseq:
            current_func = generate_inverted(Outer(current_func, k, N=N))
        self.approximation = generate_inverted(current_func)

    def __call__(self, x):
        print 'next call'
        return self.approximation(x)

    def compose(self, f, kseq):
        current_outer = f
        for k in reversed(kseq):
            current_outer = Outer(current_outer, k, self.N)
        return current_outer"""


class Combinator(object):
    def __init__(self, f_ethalon, f_norm, L, N=2, dps=2048*4):
        with workdps(dps):
            self.f = f_ethalon
            self.N = N
            self.f_norm = f_norm
            self.L = L
            kseq_estimator = KseqEstimator(f_norm, L, N, dps)
            kseq, norms = kseq_estimator.estimate_kseq(amount_of_k=3)
            print kseq
            #kseq.append(mpf('200'))
            #kseq.append(mpf('2000'))
            all_combinations = [(list(combinations(kseq, i)), (-1)**(i+1)) for i in xrange(1, len(kseq) + 1)]
            self.compositions = []
            self.signs = []
            for combos, sign in all_combinations:
                for combo in combos:
                    self.compositions.append(self.compose(f_ethalon, combo))
                    self.signs.append(sign)

    def __call__(self, x):
        #print 'next call'
        return sum([composition(x) * sign for (composition, sign) in zip(self.compositions, self.signs)])

    def compose(self, f, kseq):
        current_outer = f
        for k in reversed(kseq):
            current_outer = Outer(current_outer, k, self.N)
        return current_outer