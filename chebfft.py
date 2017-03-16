"""
Spectral methods in MATLAB. Lloyd
Program CHEBFFT
"""

# Chebyshev differentiation via FFT/ Simple, not optimal.
# If v is complex, delete "real" commands.

from numpy import *
from scipy.fftpack import dct, idct

def chebfft(v):
    N = len(v) - 1
    if (N == 0): w = 0;
    x = cos(pi*arange(0,N+1)/N)
    ii = arange(0,N)
    v = hstack([v])
    V = hstack([v, flipud(v[1:N])])
    U = real(fft.fft(V))
    W = real(fft.ifft(1j*hstack([ii, 0, arange(1-N,0)]).T*U))
    w = zeros(N+1)
    w[1:N] = -W[1:N]/sqrt(1-x[1:N]**2)
    w[0] = sum(ii.T**2*U[0:len(ii)])/float(N) + 0.5*N*U[N]
    w[N] = sum(((-1)**(ii+1))*(ii.T**2)*U[0:len(ii)])/float(N) + 0.5*(-1)**(N+1)*N*U[N]
    
    return w