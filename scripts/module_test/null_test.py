import numpy as np
import scipy
import scipy.linalg
from numpy.linalg import svd
from scipy.linalg import qr

'''
测试〇空间计算
'''


def qr_null(A, tol=None):
    Q, R, P = qr(A.T, mode='full', pivoting=True)
    tol = np.finfo(R.dtype).eps if tol is None else tol
    rnk = min(A.shape) - np.abs(np.diag(R))[::-1].searchsorted(tol)
    return Q[:, rnk:].conj()


def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def null(A, eps=1e-12):
    u, s, vh = scipy.linalg.svd(A)
    padding = max(0, np.shape(A)[1] - np.shape(s)[0])
    null_mask = np.concatenate(((s <= eps), np.ones((padding,), dtype=bool)), axis=0)
    null_space = np.compress(null_mask, vh, axis=0)
    return np.transpose(null_space)


def null_space(A, rcond=None):
    u, s, vh = svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:, :].T.conj()
    return Q


if __name__ == '__main__':
    A = np.matrix([[2, 3, 5], [-4, 2, 3]])
    print('null\n', null(A))  # <class 'numpy.ndarray'>
    print('nullspace\n', nullspace(A))  # <class 'numpy.matrix'>
    print('null_space\n', null_space(A))  # <class 'numpy.matrix'>
    print('qr_null\n', qr_null(A))  # <class 'numpy.ndarray'>
    # print(A*nullspace(A))
    a = np.matrix([1, 2, 3])
    # print(np.shape(a.T)).
    b = [[1,2,3],[2,3,4]]
    print(np.shape(b))
