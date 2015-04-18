import math
import bitstring
import numpy as np


def x_pow_decomposition(x, gamma_pow):
    #print "x: {x}".format(x=x)
    head = int(math.floor(x))
    #print "head: {h}".format(h=head)
    tail = x - head
    #print "tail: {t}".format(t=tail)

    b = bitstring.pack('>d', tail)
    nmant, nexp = np.finfo(tail).nmant, np.finfo(tail).nexp
    sbit, wbits, pbits = b[:1], b[1:1 + nexp], b[1 + nexp:]
    tail_exp = wbits.uint - (1<<(nexp - 1)) + 1
    assert tail_exp < 0
    print len(pbits.bin), tail_exp, gamma_pow
    bin_repr = '0'*(-tail_exp * gamma_pow) + '1' + pbits.bin
    print len(bin_repr)
    bin_repr += '0'*(gamma_pow - len(bin_repr) % gamma_pow)
    print len(bin_repr)
    assert len(bin_repr) % gamma_pow == 0
    compressed = [int(bin_repr[i:i+gamma_pow], 2) for i in range(0, len(bin_repr), gamma_pow)]
    compressed[0] = head
    return int(sbit.bin), np.array(compressed)


# very slow decomposition, but it works :)
def x_decomposition(x, gamma=2):
    if gamma != 2:
        if gamma & (gamma - 1) == 0:
            return x_pow_decomposition(x, gamma)
        else:
            raise Exception("gamma must be a power of two!")
    #print "x: {x}".format(x=x)
    head = int(math.floor(x))
    #print "head: {h}".format(h=head)
    tail = x - head
    #print "tail: {t}".format(t=tail)

    b = bitstring.pack('>d', tail)
    nmant, nexp = np.finfo(tail).nmant, np.finfo(tail).nexp
    sbit, wbits, pbits = b[:1], b[1:1 + nexp], b[1 + nexp:]
    tail_exp = wbits.uint - (1<<(nexp - 1)) + 1
    assert tail_exp < 0
    bin_repr = '0'*(-tail_exp) + '1' + pbits.bin
    bits = np.array([int(c) for c in bin_repr])
    powers = np.array([2**(-i) for i in xrange(bits.size)])
    assert np.sum(powers[bits == 1]) == tail
    bits[0] = head
    return int(sbit.bin), bits


def x_restore(sign, bits):
    return (-1)**sign * np.sum((bits * np.array([2**(-i) for i in xrange(bits.size)])))

# returns smallest power of 2 that is greater or equal than v
def power_2_bound(v):
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    return v + 1