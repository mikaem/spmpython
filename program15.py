"""
Spectral methods in MATLAB. Lloyd
Program 15
"""

# Solve eigenvalue BVP u_xx=lambda*u, u(-1)=u(1)=0

from numpy import *
from numpy.linalg import matrix_power,eig,solve
from matplotlib import pyplot as plt

N = 36
D,x = cheb(N)
D2 = matrix_power(D, 2)
D2 = D2[1:N,1:N]
lam,V = eig(D2)
ii = argsort(-lam)
lam = lam[ii]
V = V[:,ii]
for j in arange(5,31,5):
    u = hstack([0, V[:,j], 0])
    xx = arange(-1,1,0.01)
    uu = polyval(polyfit(x,u,N),xx)     # interpolate grid data
    plt.subplot(7,1,int(j/5.))
    plt.plot(x,u,'o', xx,uu,'-')
    plt.title('eig '+str(j)+' = '+str(lam[j]*4./pi**2)+'**4./pi**2 '+str(4.*N/(pi*j))+' ppw')
    plt.grid('on')
plt.show()