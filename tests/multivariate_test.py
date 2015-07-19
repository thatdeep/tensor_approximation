import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from function_approximation import Composer


def test_sinus():
    def ethalon_function(x):
        return math.sin(4*(x[0] + x[1]))/2

    norm = 0.5
    L = 4*math.sqrt(2.0)/2

    M = 3
    x = np.linspace(0.0, 1.0, M, endpoint=False)
    y = np.linspace(0.0, 1.0, M, endpoint=False)

    X, Y = np.meshgrid(x, y)

    points_raw = np.vstack([X.ravel(), Y.ravel()]).T
    values_raw = np.apply_along_axis(ethalon_function, 1, points_raw)
    V = values_raw.reshape(X.shape)

     # Twice as wide as it is tall.`
    fig = plt.figure(figsize=plt.figaspect(1.0))
    #---- First subplot
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    surf = ax.plot_surface(X, Y, V, rstride=1, cstride=1, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)
    ax.set_zlim3d(V.min(), V.max())
    fig.colorbar(surf, shrink=0.5, aspect=10)

    for i in xrange(1, 4):
        combinator = Composer(ethalon_function, norm, L, N=2, initial_k=1, amount_of_k=i)
        comb_values_raw = [combinator(point) for point in points_raw]
        CV = np.array(comb_values_raw, dtype=np.float).reshape(X.shape)
        print "k={k}: max_err: {me}".format(k=i, me=np.max(np.abs(CV - V)))

        #---- Second subplot
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        surf = ax.plot_surface(X, Y, CV, rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=10)
        ax.set_zlim3d(CV.min(), CV.max())
    plt.savefig("test.png", dpi=300)
    plt.show()

    #for k in xrange(1, 10):
    #    combinator = Composer(ethalon_function, norm, L, N=2, initial_k=k)
    #    comb_values_raw = [combinator(point) for point in points_raw]
    #    CV = np.array(comb_values_raw, dtype=np.float).reshape(X.shape)
    #    print ('{k}\t{max_err}\t{err_norm}\n'+'*'*80).format(k=k, max_err=np.max(np.abs(V - CV)), err_norm=np.linalg.norm(V - CV))

def test_sinus_write(M=20):
    def ethalon_function(x):
        return math.sin(4*(x[0] + x[1]))/2

    norm = 0.5
    L = 4*math.sqrt(2.0)/2
    x = np.linspace(0.0, 1.0, M, endpoint=False)
    y = np.linspace(0.0, 1.0, M, endpoint=False)

    X, Y = np.meshgrid(x, y)

    points_raw = np.vstack([X.ravel(), Y.ravel()]).T
    values_raw = np.apply_along_axis(ethalon_function, 1, points_raw)
    V = values_raw.reshape(X.shape)

    CC = np.zeros((4, values_raw.size))
    CC[0] = values_raw

    for i in xrange(1, 4):
        combinator = Composer(ethalon_function, norm, L, N=2, initial_k=1, amount_of_k=i)
        comb_values_raw = [combinator(point) for point in points_raw]
        CV = np.array(comb_values_raw, dtype=np.float).reshape(X.shape)
        CC[i] = comb_values_raw
        print "k={k}: max_err: {me}".format(k=i, me=np.max(np.abs(CV - V)))
    CC.tofile("saved_{M}".format(M=M))