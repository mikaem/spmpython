"""
Spectral methods in MATLAB. Lloyd
Program 11
"""

# Chebyshev differentiation of a smooth function

from numpy import *
from matplotlib import pyplot as plt

xx = arange(-1,1,0.01)
uu = exp(xx)*sin(5*xx)
for N in [10, 20]:
    D,x = cheb(N)
    u = exp(x)*sin(5*x)
    
    plt.subplot(220+int(N/10.))
    plt.plot(x, u, marker='o', linestyle='-')
    plt.title('f(x), N=' + str(N))
    
    error = D.dot(u)-exp(x)*(sin(5*x)+5*cos(5*x))
    plt.subplot(222+int(N/10.))
    plt.plot(x,error, marker='o', linestyle='-')
    plt.title('error in f''(x), N=' + str(N))
    
plt.show()