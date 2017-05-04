"""
Spectral methods in MATLAB. Lloyd
Program 32
"""

# Solve u_xx=exp(4x), u(-1)=0, u(1)=1 (compare program 13)

from numpy import *
from cheb import cheb
from numpy.polynomial import chebyshev as n_cheb
from numpy.linalg import matrix_power,solve,norm
from matplotlib import pyplot as plt

N = 16
D,x = cheb(N)
D2 = matrix_power(D, 2)
D2 = D2[1:N,1:N]                 # boundary conditions
f = exp(4*x[1:N])
u = solve(D2,f)                    # Poisson equation solved here
u = hstack([0, u, 0]) + (x+1)/2.0
xx = arange(-1,1,0.01)
#uu = polyval(polyfit(x,u,N),xx)     # interpolate grid data
uh = n_cheb.chebfit(x, u, N)
uhx = n_cheb.chebval(xx, uh)
plt.plot(x,u,'o', xx,uhx,'-')
exact = (exp(4*xx)-sinh(4)*xx-cosh(4))/16. + (xx+1)/2.0
plt.title('max err = ' + str(norm(uhx-exact,inf)))
plt.grid('on')

plt.show()