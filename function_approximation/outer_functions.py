import numpy as np

from numpy import float128 as f128

import decimal as dc

def dot_sequence(indices, gamma, eps, k_max=15):
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
    return dots


