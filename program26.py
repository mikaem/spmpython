"""
Spectral methods in MATLAB. Lloyd
Program 26
"""

# Eigenvalues of 2nd-order Chebyshev diff.matrix

from numpy import *
from scipy.linalg import eig
from matplotlib import pyplot as plt
from cheb import cheb
from numpy.linalg import matrix_power
from numpy.polynomial import chebyshev as n_cheb

N = 60
[D,x] = cheb(N)
D2 = matrix_power(D, 2)
D2 = D2[1:N,1:N]
Lam, V = eig(D2)
ii = argsort(-Lam)
e = Lam[ii]
V = V[:,ii]

# Plot eigenvalues
plt.subplot(3,1,1)
plt.hold('on')
plt.loglog(-real(e),'o')
plt.ylabel('eigenvalue')
plt.title('N = '+str(N)+', max|lambda| = '+str(max(-e)/N**4)+' N^4')
plt.semilogy([2*N/pi,2*N/pi],[1,1e6],'--r')
plt.text(2.1*N/pi, 24, '2*Pi/N')

# Plot eigenmodes N/4 (physical) and N (nonfysical):
vN4 = hstack([0,V[:,int(N/4.0)-2],0])
xx = arange(-1,1.01,0.01)
vh = n_cheb.chebfit(x, vN4, N)
vv = n_cheb.chebval(xx, vh)
plt.subplot(3,1,2)
plt.hold('on')
plt.plot(xx,vv)
plt.plot(x,vN4,'o')
plt.title('eigenmode N/4')
plt.axis([-1,1,-0.2,0.2])
vN = hstack([0,V[:,N-2],0])
plt.subplot(3,1,3)
plt.hold('on')
plt.semilogy(x,abs(vN))
plt.axis([-1,1,-1,1])
plt.plot(x,vN,'o')
plt.title('modulus of eigenmode N    (log scale)')
plt.show()

