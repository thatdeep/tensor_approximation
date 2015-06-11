import math
import numpy as np
import mpmath as mp

from utils import power_2_bound, gamma_estimate
from mpmath import workdps, mpf


class KseqEstimator(object):
    def __init__(self, norm, L, N=2, dps=2048*4, initial_k=1):
        with workdps(dps):
            param = mpf('1.0')
            self.L = L
            self.N = 2
            self.gamma = mpf(gamma_estimate(N))
            self.m = mpf(self.N)
            self.norm = norm
            self.dps = dps
            self.eps_coeff = (self.gamma - 3) / (self.gamma - 1)
            self.sconst = (param * mpf('0.5')*(self.gamma - 3) / (self.gamma - 1))**(mp.ln(2) / mp.ln(self.gamma))
            self.initial_k = initial_k

    def nth(self, n):
        with workdps(self.dps):
            if n == 0: return mpf(0)
            return ((self.m+2) * self.m**(n-1) - 3) / (self.m - 1)

    def epsilon(self, k):
        with workdps(self.dps):
            return self.eps_coeff * self.gamma**(-k)

    # find_minimum_k
    def find_minimum_k(self, k_last=None):
        with workdps(self.dps):
            #print 'last_k:', k_last
            # delta func
            delta_f = lambda k: (self.gamma - 2) / (self.gamma - 1) * self.gamma**(-k)
            #delta_f = lambda k: (self.gamma - 2) * self.gamma**(-self.nth(k + 1))

            if k_last == None:
                return mpf(self.initial_k)
            assert k_last > 0
            c_bound = self.sconst
            # TODO Ek definition is different!!!
            #print 'maxmax:', mp.ceil(-mp.log(delta_f(k_last)/c_bound, 2)), k_last + 1
            return max(mp.ceil(-mp.log(delta_f(k_last)/c_bound, 2)), k_last + 1)

    def search_optimal_k(self, inequality, low, high):
        with workdps(self.dps):
            while True:
                if high == low:
                    return low
                if high == low + 1:
                    if inequality(self.epsilon(low) * self.L, low):
                        return low
                    else:
                        return high
                middle = mp.floor((high + low) / 2)
                modulo = self.epsilon(middle) * self.L
                if inequality(modulo, middle):
                    high = middle
                else:
                    low = middle + 1

    def find_optimal_k(self, inequality, k_last=None):
        with workdps(self.dps):
            k_next = self.find_minimum_k(k_last)
            valid = k_next
            #if k_last != None and mp.log(self.gamma, 2) * self.nth(k_last) > self.dps * mp.log(mpf('10'), mpf('2')):
            #    return mp.log(self.gamma, 2) * self.nth(k_last), -1
            i = 1
            while True:
                modulo = self.epsilon(k_next) * self.L
                if inequality(modulo, k_next):
                    k_returned = self.search_optimal_k(inequality, valid, k_next)
                    return self.epsilon(k_returned) * self.L, k_returned
                #print 'k_reached:', k_next
                k_next *= 2**i
                i += 1

    def estimate_kseq(self, amount_of_k=3):
        beta_map = {}
        max_size = amount_of_k + 1
        with workdps(self.dps):
            # find proper gamma for N
            power_of_two = power_2_bound(2*self.N + 3)
            gamma_pow = int(math.log(power_of_two, 2))
            gamma = mpf(power_of_two)

            # define a_value function
            A_CONST = mpf(self.sconst) * (2*self.N + 1) / (self.N + 1)
            a_value_empty = lambda r: A_CONST * gamma**beta_map[kseq[r-1]+1] * mpf(2)
            a_value = lambda n, r: A_CONST * gamma**beta_map[kseq[r-1]+1] * mpf(2)**(1 - kseq[n-1])

            estimated_norm = self.norm
            bseq = [estimated_norm]

            # First iteration - very simple
            inequality = lambda modulo, k: modulo <= estimated_norm / mpf(2*self.N + 2)
            b_one, k_one = self.find_optimal_k(inequality)
            kseq = [k_one]
            beta_map[k_one+1] = self.nth(k_one + 1)
            bseq.append(b_one)

            # Now, make a_nr and c_nj sequences
            c_nj = [[0]*max_size for _ in xrange(max_size)]
            a_nr = [[0]*max_size for _ in xrange(max_size)]
            for i in xrange(max_size):
                a_nr[i][i] = mpf(self.N) / mpf(self.N + 1)
                a_nr[i][0] = mpf('1')
                c_nj[i][i] = mpf('1')
                for j in xrange(i + 1, max_size):
                    a_nr[i][j] = None

            #a_nr[1][0] = a_value(1, 0)
            c_nj[1][0] = a_nr[1][1] * c_nj[0][0]

            partial_sums = []

            # Next iterations
            for t in xrange(2, max_size):
                j_max = t - 1
                for r in xrange(1, t):
                    a_nr[t][r] = a_value_empty(r)
                ssum = sum([c_nj[j_max - 1][k] * bseq[k] for k in xrange(j_max)])
                partial_sums.append(a_nr[t][j_max] * ssum)
                second_part = sum([c_nj[j_max][j] * bseq[j] for j in xrange(t)]) / (2*self.N + 2)
                #print 'SP:', second_part
                inequality = lambda modulo, k: (modulo + sum(partial_sums) / (2**(k))) <= second_part
                b_next, k_next = self.find_optimal_k(inequality, k_last=kseq[-1])
                if k_next == -1:
                    #print 'reach power of', b_next
                    return kseq, bseq
                kseq.append(k_next)
                bseq.append(b_next)
                beta_map[k_next + 1] = self.nth(k_next + 1)
                #print b_next, k_next

                for r in xrange(1, t):
                    a_nr[t][r] /= 2**(k_next)
                for j in xrange(t - 1, -1, -1):
                    c_nj[t][j] = sum([a_nr[t][k] * c_nj[k-1][j] for k in xrange(j+1, t + 1)])
            return kseq, bseq