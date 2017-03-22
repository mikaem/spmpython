"""
Spectral methods in MATLAB. Lloyd
Program Program 21
"""

# Eigenvalues of Mathieu operator -u_xx+2qcos(2x)u (compare program 8)

from numpy import *
from scipy.linalg import toeplitz
from matplotlib import pyplot as plt

N = 42
h = 2*pi/N
x = h*arange(1,N+1)
col = hstack([-pi**2/(3*h**2)-1./6, -0.5*(-1)**arange(1,N)/sin(h*arange(1,N)/2)**2])
D2 = toeplitz(col,col)
qq = arange(0,15.2,0.2)
data = zeros((len(qq),11))

for i in range(len(qq)):
    e, v = linalg.eig(-D2+2*qq[i]*diag(cos(2*x)))
    eg = sorted(e) 
    data[i,:] = hstack([eg[1:12]])

plt.plot(qq,data)
plt.xlabel('q')
plt.ylabel('lambda')
plt.axis([0, 15, -24, 32])
plt.show()
