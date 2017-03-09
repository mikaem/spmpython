"""
Spectral methods in MATLAB. Lloyd
Program 14
"""

# Solve nonlinear BVP u_xx=exp(u), u(-1)=u(1)=0 (compare program 13)

from numpy import *
from numpy.linalg import matrix_power,solve,norm
from matplotlib import pyplot as plt

N = 16
D,x = cheb(N)
D2 = matrix_power(D, 2)
D2 = D2[1:N,1:N]
u = zeros(N-1)
change = 1; it = 0
while (change > 1e-15):
    unew = solve(D2,exp(u))
    change = norm(unew-u,inf)
    u = unew; it += 1;
u = hstack([0, u, 0])
xx = arange(-1,1,0.01)
uu = polyval(polyfit(x,u,N),xx)     # interpolate grid data
plt.plot(x,u,'o', xx,uu,'-')
plt.title('no. steps = ' + str(it) + ' u(0) = ' + str(u[N/2.+1]))
plt.grid('on')
plt.show()