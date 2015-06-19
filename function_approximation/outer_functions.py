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
            return [([mp.floor(el + eps * q * self.gamma**self.k) for el in x_big], q) for q in qs], qs_mask

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
                    #multipliers[i] = [mpf('0.5'), mpf('0.5')]
                    multipliers[i] = [(indices[i][0] + eps - (x_big[i] + mod_val))/eps, (x_big[i] + mod_val - indices[i][0])/eps]

            # Now we have indices, where indices[i] may be number, or a pair of numbers (if x[i] lay between intervals)
            return indices, multipliers


class SemiOuter(object):
    def __init__(self, f, k, N=2, dps=2048*4):
        with workdps(dps):
            self.f = f
            self.k = k
            self.dps = dps
            self.N = N
            self.gamma = mpf(gamma_estimate(N))
            #if math.log(10, self.gamma) * k > dps:
            #    self.dps = math.log(10, self.gamma) * k * 3
            self.cube_dispatcher = CubeDispatcher(k, N, self.dps)
            self.mul_const = self.gamma**(-self.k)
            self.eps = mpf('1.0') / (self.gamma - 1)
            real_dps = math.log(self.gamma, 10)*k
            #if 4*real_dps > dps:
            #    self.dps = max(4*real_dps, dps)
            eps_mul_const = self.eps * self.mul_const
            #self.all_eps_adders = [eps_mul_const - q * self.eps for q in xrange(2 * N + 1)]

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
            return self.f(x) - sum_for_q

    def average_between_cubes(self, indices, multipliers, q):
        with workdps(self.dps):
            eps = mpf('1.0') / (self.gamma - 1)
            ind = [[i] if type(i) != list else i for i in indices]
            mlt = [[m] if type(m) != list else m for m in multipliers]
            shp = [len(i) for i in ind]
            f_value = mpf('0.0')
            #mul_const = self.gamma**(-self.k)
            m = int(np.prod(shp))
            for i in np.ndindex(*shp):
                point = [(ind[s][i[s]] + eps) * self.mul_const - eps*q for s in xrange(self.N)]
                #print point
                f_value += mpf(self.f(point)) * np.prod([mlt[s][i[s]] for s in xrange(self.N)])
            f_value /= (self.N + 1)
            return f_value

    def exact_on_cube(self, index, q):
        with workdps(self.dps):
            eps = mpf('1.0') / (self.gamma - 1)
            #mul_const = self.gamma**(-self.k)
            point = [(index[s] + eps) * self.mul_const - q * eps for s in xrange(self.N)]
            #point = [indices[s]*self.mul_const + self.all_eps_adders[q] for s in xrange(self.N)]
            return mpf(self.f(point)) / (self.N + 1)


class Composer(object):
    def __init__(self, f_ethalon, f_norm, L, N=2, dps=2048*4, initial_k=1, amount_of_k=1):
        with workdps(dps):
            self.f = f_ethalon
            self.N = N
            self.f_norm = f_norm
            self.L = L
            kseq_estimator = KseqEstimator(f_norm, L, N, dps, initial_k=initial_k)
            self.kseq, self.norms = kseq_estimator.estimate_kseq(amount_of_k=amount_of_k)
            print self.kseq
            self.next_semi_outer = SemiOuter(f_ethalon, k=self.kseq[0], N=N, dps=dps)
            for k in xrange(1, len(self.kseq)):
                self.next_semi_outer = SemiOuter(self.next_semi_outer, k=self.kseq[k], N=N, dps=dps)

    def __call__(self, x):
        return self.f(x) - self.next_semi_outer(x)