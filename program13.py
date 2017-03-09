"""
Spectral methods in MATLAB. Lloyd
Program 13
"""

# Solve linear BVP u_xx=exp(4x), u(-1)=u(1)=0

from numpy import *
from numpy.linalg import matrix_power,solve,norm
from matplotlib import pyplot as plt

N = 16
D,x = cheb(N)
D2 = matrix_power(D, 2)
D2 = D2[1:N,1:N]                 # boundary conditions
f = exp(4*x[1:N])
u = solve(D2,f)                    # Poisson equation solved here
u = hstack([0, u, 0])
xx = arange(-1,1,0.01)
uu = polyval(polyfit(x,u,N),xx)     # interpolate grid data
plt.plot(x,u,'o', xx,uu,'-')
exact = (exp(4*xx)-sinh(4)*xx-cosh(4))/16.
plt.title('max err = ' + str(norm(uu-exact,inf)))
plt.grid('on')
plt.show()