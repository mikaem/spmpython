"""
Spectral methods in MATLAB. Lloyd
Program CHEBFFT
"""

# Chebyshev differentiation via FFT/ Simple, not optimal.
# If v is complex, delete "real" commands.

from numpy import *
from scipy.fftpack import dct, idct, idst, dst

def chebfft(v):
    N = len(v) - 1
    if (N == 0): w = 0;
    x = cos(pi*arange(0,N+1)/N)
    K = fft.fftfreq(2*N, 0.5/N)
    K[N] = 0
    ii = arange(0,N)
    v = hstack([v])
    V = hstack([v, flipud(v[1:N])])
    U = real(fft.fft(V))
    uu = dct(v, 1)
    uv = hstack((uu, uu[1:-1][::-1]))
    W = real(fft.ifft(1j*K*U))
    ww = dst(K*uv, 1)
    from IPython import embed; embed()
    w = zeros(N+1)
    w[1:N] = -W[1:N]/sqrt(1-x[1:N]**2)
    w[0] = sum(ii.T**2*U[0:len(ii)])/float(N) + 0.5*N*U[N]
    w[N] = sum(((-1)**(ii+1))*(ii.T**2)*U[0:len(ii)])/float(N) + 0.5*(-1)**(N+1)*N*U[N]

    return w

N = 8
x = cos(pi*arange(0,N+1)/N)
v = exp(-4*x)
chebfft(v)
