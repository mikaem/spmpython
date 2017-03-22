"""
Spectral methods in MATLAB. Lloyd
Program Program 24
"""

# Pseudospectra of Davies' complex harmonic oscillator
# (For finer, slower plot, change 0:2 to 0:0.5)

from numpy import *
from scipy.linalg import eig
from matplotlib import pyplot as plt
from cheb import cheb
from numpy.linalg import matrix_power, svd


# Eigenvalues

N = 70
[D,x] = cheb(N)
x = x[1:N]
L = 6; x = L*x; D = D/L;           #rescale to [-L,L]
A = -matrix_power(D, 2)
A = A[1:N,1:N] + (1+3j)*diag(x**2)
E,V = eig(A)
ii = argsort(E)
#E = E[ii]
plt.hold('on')
plt.plot(E,'o')
plt.axis([0,50,0,40])

# Pseudospectra

x = arange(0,52,2)
y = arange(0,42,2)
[xx,yy] = meshgrid(x,y)
zz = xx +1j*yy
I = identity(N-1)
sigmin = zeros((len(y),len(x)))
for j in range(0,len(x)):
    for i in range(0,len(y)):
        sigmin[i,j] = min(svd(zz[i,j]*I-A)[1])
plt.contour(xx,yy,sigmin,10**arange(-4,0,0.5))
plt.show()
