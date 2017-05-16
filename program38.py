"""
Spectral methods in MATLAB. Lloyd
Program 38
"""

# Solve u_xxxx=exp(x), u(-1)=u(1)=u'(-1)=u'(1)=0  (compare pr.13)

from numpy import *
from numpy.linalg import matrix_power,solve,norm
from matplotlib import pyplot as plt
from cheb import cheb
from numpy.polynomial import chebyshev as n_cheb


# Construct discrete biharmonic operator
N = 15
D,x = cheb(N)
S = diag(hstack([0, 1.0/(1-x[1:N]**2),0]))
D4 = (diag(1-x**2).dot(matrix_power(D,4)) - 8*diag(x).dot(matrix_power(D,3)) - 12*matrix_power(D,2)).dot(S)
D4 = D4[1:N,1:N]                 # boundary conditions

# Solve boundary-value problem and plot result:
f = exp(x[1:N])
u = solve(D4,f)                    # Poisson equation solved here
u = hstack([0, u, 0])
xx = arange(-1,1.01,0.01)
uh = n_cheb.chebfit(x, S.dot(u), N)
uhx = (1-xx**2)*n_cheb.chebval(xx, uh)
plt.plot(x,u,'o', xx,uhx,'-')
plt.axis([-1, 1,-0.01,0.06])

# Determine exact solution and print maximum error:
A = [[1,-1,1,-1],[0,1,-2,3],[1,1,1,1],[0,1,2,3]]
V = vander(xx)
V = V[:,-1:-5:-1]
c = solve(A,exp([-1,-1,1,1]))
exact = exp(xx)-V.dot(c)
plt.title('max err = ' + str(norm(uhx-exact,inf)))
plt.grid('on')
plt.show()