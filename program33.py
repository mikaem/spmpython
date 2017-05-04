"""
Spectral methods in MATLAB. Lloyd
Program 33
"""

# Solve u_xx=exp(4x), u'(-1)=u(1)=0

from numpy import *
from cheb import cheb
from numpy.polynomial import chebyshev as n_cheb
from numpy.linalg import matrix_power,solve,norm
from matplotlib import pyplot as plt

N = 16
D,x = cheb(N)
D2 = matrix_power(D, 2)
D2[N,:] = D[N,:]                 # Neumann condition at x = -1
D2 = D2[1:N+1,1:N+1]                 # boundary conditions
f = exp(4*x[1:N])
u = solve(D2,hstack([f,0]))                    # Poisson equation solved here
u = hstack([0, u])
xx = arange(-1,1,0.01)
#uu = polyval(polyfit(x,u,N),xx)     # interpolate grid data
uh = n_cheb.chebfit(x, u, N)
uhx = n_cheb.chebval(xx, uh)
plt.plot(x,u,'o', xx,uhx,'-')
exact = (exp(4*xx)-4*exp(-4)*(xx-1)-exp(4))/16.
plt.title('max err = ' + str(norm(uhx-exact,inf)))
plt.grid('on')

plt.show()