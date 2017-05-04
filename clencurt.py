"""
Spectral methods in MATLAB. Lloyd
Program CLENCURT
"""

# CLENCURT nodes x (Chebyshev points) and weights w for Clenshaw-Curtis quadrature

from numpy import *

def clencurt(N):
    theta = pi*arange(0,N+1)/N
    x = cos(theta)
    w = zeros(N+1)
    ii = arange(1,N)
    v = ones(N-1)
    if mod(N,2)==0:
        w[0] = 1./(N**2-1)
        w[N] = w[0]
        for k in arange(1,int(N/2.)):
            v = v-2*cos(2*k*theta[ii])/(4*k**2-1)
        v = v - cos(N*theta[ii])/(N**2-1)
    else:
        w[0] = 1./N**2
        w[N] = w[0]
        for k in arange(1,int((N-1)/2.)+1):
            v = v-2*cos(2*k*theta[ii])/(4*k**2-1)
    w[ii] = 2.0*v/N
    return x, w