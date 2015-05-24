from function_approximation import KseqEstimator

def simple_kseq_test(norm, L, N=2):
    kseq_estimator = KseqEstimator(norm, L, N=N)
    print kseq_estimator.k_finder(4)