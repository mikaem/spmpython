"""
Spectral methods in MATLAB. Lloyd
Program CHEB
"""

# CHEB compute D = differentiation matrix, x = Chebyshev grid

from numpy import *
from scipy.sparse import diags
from scipy.linalg import toeplitz

def cheb(N):
    if (N == 0):
        D = 0
        x = 1
    x = cos(pi*arange(0,N+1)/N)
    c = hstack([2, ones(N-1), 2])*(-1)**arange(0,N+1)
    X = tile(x,(N+1,1))
    dX = X.T - X
    D = (c[:,newaxis]*(1.0/c)[newaxis,:])/(dX+(identity(N+1)))       # off-diagonal entries
    D = D - diag(D.sum(axis=1))              # diagonal entries
    return D, x
    