"""
Spectral methods in MATLAB. Lloyd
Program 18a
"""

# Chebyshev differentiation via FFT (compare program 11)

from numpy import *
from matplotlib import pyplot as plt
from chebfft import chebfft

xx = arange(-1,1,0.01)
ff = exp(xx)*sin(5*xx)
for N in [10, 20]:
    x = cos(pi*arange(0,N+1)/N)
    f = exp(x)*sin(5*x)
    
    plt.subplot(220+int(N/10.))
    plt.plot(x, f, 'o', xx,ff,'-')
    plt.title('f(x), N=' + str(N))
    
    error = chebfft(f)-exp(x)*(sin(5*x)+5*cos(5*x))
    plt.subplot(222+int(N/10.))
    plt.plot(x,error, marker='o', linestyle='-')
    plt.title('error in df(x)/dx, N=' + str(N))
    
plt.show()