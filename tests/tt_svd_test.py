import numpy as np
from tensor_train.core import TensorTrain
from tensor_train.utils import frobenius_norm


percent = 0.98
A = np.random.random((5, 5, 5, 5))
B = np.random.random((5, 5, 5, 5))
zero_idx = np.array(np.random.binomial(1, percent, A.shape), dtype=np.bool)
A[zero_idx] = 0
zero_idx = np.array(np.random.binomial(1, percent, A.shape), dtype=np.bool)
B[zero_idx] = 0

#print A
print frobenius_norm(A)
print '---'

eps = 1e-16
tA = TensorTrain(A, eps)
tB = TensorTrain(B, eps)
AA = tA.full_tensor()
BB = tB.full_tensor()
print '---'
#print AA
print 'in TA'
print [cr.size for cr in tA.cores]
print [cr.shape for cr in tA.cores]
print [rk for rk in tA.r]

print '---'
print 'in TB'
print [cr.size for cr in tB.cores]
print [cr.shape for cr in tB.cores]
print [rk for rk in tB.r]

print '---'
print 'in TC'
tC = tA + tB
print [cr.size for cr in tC.cores]
print [cr.shape for cr in tC.cores]
print [rk for rk in tC.r]
print '-*-*-'
print np.linalg.norm((tC.full_tensor() - (AA - BB)).reshape(A.size)), eps * np.linalg.norm(A.reshape(A.size))

tCC = tC.tt_round(eps)
print [cr.size for cr in tCC.cores]
print [cr.shape for cr in tCC.cores]
print [rk for rk in tCC.r]
print '-*-*-'
print np.linalg.norm((tCC.full_tensor() - (AA - BB)).reshape(A.size)), eps * np.linalg.norm(A.reshape(A.size))


#B = t.tt_round(eps)
#BB = B.full_tensor()
#print np.linalg.norm((AA - BB).reshape(A.size)), eps * np.linalg.norm(A.reshape(A.size))

print '-*-*-'
tAN = -tA
print np.linalg.norm((tAN.full_tensor() + AA).reshape(A.size)), eps * np.linalg.norm(A.reshape(A.size))

print tA.convolve(np.ones((4, 5)))
print tA.scalar_product(tB)

print np.linalg.norm(((tA * tB).full_tensor() - AA*BB).reshape(A.size)), eps * np.linalg.norm(A.reshape(A.size))