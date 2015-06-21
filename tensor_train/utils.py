import numpy as np

from core import TensorTrain


def frobenius_norm(a):
    """
    Compute Frobenius norm for tensor a.

    Parameters
    ----------
    a : {(..., N1, ..., Nd) array_like, TensorTrain}
        input tensor

    Returns
    -------
    norm : float
        frobenius norm of a

    Raises
    ------
    Exception
        if a is not ndarray or TensorTrain
    """
    if isinstance(a, TensorTrain):
        if a.normed():
            return a.norm
        else:
            return a.tt_frobenius_norm()
    if isinstance(a, np.ndarray):
        return np.linalg.norm(a)
    else:
        raise Exception('Wrong type of argument - must be ndarray of TensorTrain!')


def rank_chop(s, delta):
    """
    Clip vector s = [\sigma_{1}, \ldots, \sigma{r}] of real values sorted in descending order
    as s without \sigma_{r'}, \sigma_{r'+1} \ldots, \sigma_{r}
    such that \sum\limits_{i=r'}^{r} \sigma_{i} \geqslant \delta

    Parameters
    ----------
    s : (..., r) array_like
        A real-valued array of size r
    delta : float, optional
        clipping parameter. Zero by default, so, if not specified, will cause nothing.

    Returns
    -------
    sr : (..., r') array_like
        Clipped array
    """
    error = 0
    i = 1
    while i <= s.size and error + s[-i] < delta:
        error += s[-i]
        i += 1
    return s.size - i + 1


def csvd(a, delta=0):
    """
    Approximately factors matrix 'a' as u * np.diag(s) * v
    as SVD without \sigma_{r'}, \sigma_{r'+1} \ldots, \sigma_{r}
    such that \sum\limits_{i=r'}^{r} \sigma_{i} \geqslant \delta

    See also numpy.linalg.svd

    Parameters
    ----------
    a : (..., M, N) array_like
        A real or complex matrix of shape ('M', 'N')
    delta : float, optional
        clipping parameter. Zero by default, so, if not specified, will cause
        ordinary svd execution with economic matrices

    Returns
    -------
    u : (..., M, r') array
        Unitary matrix
    s : (..., K) array
        The clipped singular values for every matrix, sorted in descending order
    v : (..., r', N) array
        Unitary matrix

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.
    """
    u, s, v = np.linalg.svd(a, full_matrices=False)
    r_new = rank_chop(s, delta)
    return u[:, :r_new], s[:r_new], v[:r_new]