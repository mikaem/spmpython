"""
Spectral methods in MATLAB. Lloyd
Program GAUSS
"""

# GAUSS nodes x (Legendre points) and weights for Gauss quadrature

from numpy import *
from numpy.linalg import eig

def gauss(N):
    beta = 0.5/sqrt(1.-(2.*arange(1,N))**(-2))
    T = diag(beta, 1) + diag(beta, -1)
    x, V = eig(T)
    i = argsort(x)
    x = x[i]
    w = 2*V[0,i]**2
    return x, w
