import numpy as np
import math

from utils import decompose, rfunc_builder, power_2_bound
import decimal as dc


class InnerFunction:
    def __init__(self, N=2, gamma=None, dec_precision=2048):
        self.N = N
        pow_2 = power_2_bound(2*N + 3)
        dc.getcontext().prec=dec_precision
        if gamma:
            self.gamma = max(power_2_bound(gamma), pow_2)
        else:
            self.gamma = pow_2
        self.gamma_pow = int(math.log(self.gamma, 2))
        self.gamma = dc.Decimal(self.gamma)
        self.dec_precision = dec_precision
        self.betas = self.list_of_betas()
        self.max_precision = len(self.betas) - 2

    def evaluate(self, x):
        bits = decompose(math.fabs(x), self.gamma_pow, self.max_precision)
        ps = self.representation(bits)
        gammas = [dc.Decimal(self.gamma)]*bits.size
        #print bits.size, len(ps), len(self.betas)
        return np.sum([c * gamma **(-power) for (c, gamma, power) in zip(ps, gammas, self.betas[:bits.size])])


    def list_of_betas(self):
        max_beta = int(self.dec_precision * math.log(10, 2)) / self.gamma_pow
        dc.getcontext().prec=self.dec_precision
        betas = [dc.Decimal(0), dc.Decimal(1)]
        constant = dc.Decimal(2)
        for k in xrange(2, self.dec_precision, 1):
            new_bk = betas[-1] * self.N + constant + 1
            if new_bk > max_beta:
                return betas
            #assert self.gamma ** (-new_bk) < self.gamma**(-self.N * betas[-1] - constant)
            betas.append(new_bk)
        return betas

    def representation(self, bits):
        rfunc = rfunc_builder(2**self.gamma_pow, bits).astype(int)
        simple_kron = (bits == (2**self.gamma_pow - 1)).astype(int)
        max_k = bits.size
        dc.getcontext().prec = self.dec_precision
        ms = [0]*max_k
        ps = [0]*max_k
        ms[0] = dc.Decimal(self.gamma)
        ps[0] = dc.Decimal(bits[0])
        #print bits.size, len(self.betas), max_k
        for k in xrange(1, max_k):
            ms[k] = self.gamma ** (self.betas[k+1] - self.betas[k]) * ((ms[k-1] - self.gamma) * rfunc[k] / 2 + 1)
            ps[k] = int(simple_kron[k]) * (ms[k-1] + self.gamma - 2) / 2 + dc.Decimal(bits[k])*(1 - int(simple_kron[k]))
        return ps