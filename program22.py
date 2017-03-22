"""
Spectral methods in MATLAB. Lloyd
Program Program 22
"""

# 5th eigenvector of Airy equation u_xx = lambda*x*u

from numpy import *
from scipy.special import airy
from scipy.linalg import eig
from matplotlib import pyplot as plt
from cheb import cheb
from numpy.polynomial import chebyshev as n_cheb
from numpy.linalg import matrix_power

for N in arange(12,49,12):
    [D,x] = cheb(N)
    D2 = matrix_power(D, 2)
    D2 = D2[1:N,1:N]
    Lam, V = eig(D2,diag(x[1:N]))
    ii = [i for (i,lam) in enumerate(Lam) if (lam>0)]
    V = V[:,ii]
    Lam = Lam[ii]
    ii = argsort(Lam)
    ii = ii[4]
    lam = Lam[ii]
    v = hstack([0, V[:,ii], 0])
    v = v/v[int(N/2.)+1]*airy(0)[0]
    xx = arange(-1,1.01,0.01)
    uh = n_cheb.chebfit(x, v, N)
    uhx = n_cheb.chebval(xx, uh)
    plt.subplot(2,2,int(N/12.))
    plt.plot(xx, uhx, 'k')
    plt.title('N = '+str(N)+' eig = '+str(lam))
    plt.grid('on')
    plt.xlim((-1,1))
plt.show()
