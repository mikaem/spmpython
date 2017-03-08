"""
Spectral methods in MATLAB. Lloyd
Program 12
"""

# Accuracy of Chebyshev spectral differentiation (compare program 7)

from numpy import *
from scipy.linalg import norm
from matplotlib import pyplot as plt

# Compute derivatives for various values of N:
Nmax = 50
E = zeros((4,Nmax))
for N in range(1,Nmax+1):
    D,x = cheb(N)
    v = abs(x)**3
    vprime = 3*x*abs(x)
    E[0,N-1] = norm(D.dot(v)-vprime, inf)
    v = exp(-x**(-2))
    vprime = 2.*v/x**3
    E[1,N-1] = norm(D.dot(v)-vprime, inf)
    v = 1./(1+x**2)
    vprime = -2*x*v**2
    E[2,N-1] = norm(D.dot(v)-vprime, inf)
    v = x**10
    vprime = 10*x**9
    E[3,N-1]= norm(D.dot(v)-vprime, inf)

# Plot results
titles=['|x^3|', 'exp(-x^{-2})', '1/(1+x^2)', 'x^(10)']   
for iplot in range(0,4):
    plt.subplot(2,2,iplot+1)
    plt.semilogy(arange(1,Nmax), E[iplot,arange(1,Nmax)], marker='o', linestyle='-')
    plt.axis([0, Nmax, 1.1e-16, 1.1e3])
    plt.title(titles[iplot])
    plt.xlabel('N')
    plt.ylabel('error')
        
plt.show()