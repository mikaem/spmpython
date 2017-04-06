"""
Spectral methods in MATLAB. Lloyd
Program 29
"""

# Solve Poisson equation on the unit disk (compare program 16 and 28)

from numpy import *
from scipy.linalg import toeplitz
from matplotlib import pyplot as plt
from cheb import cheb
from numpy.polynomial import chebyshev as n_cheb
from numpy.linalg import matrix_power, solve
from mpl_toolkits.mplot3d import axes3d

# Laplacian in polar coordinates:

N = 31
N2 = int((N-1)/2.0)
[D,r] = cheb(N)
D2 = matrix_power(D, 2)
D1 = D2[1:N2+1,1:N2+1]
D2 = D2[1:N2+1,N-1:N2:-1]
E1 = D[1:N2+1,1:N2+1]
E2 = D[1:N2+1,N-1:N2:-1]

# t = theta coordinate, ranging from 0 to 2*pi (M must be even):
M = 40
dt = 2*pi/M
t = dt*arange(1,M+1)
M2 = int(M/2.0)
col = hstack([-pi**2/(3*dt**2)-1./6, 0.5*(-1)**arange(2,M+1)/sin(dt*arange(1,M)/2)**2])
D2t = toeplitz(col,col)

# Laplacian in polar coordinates:
R = diag(1.0/r[1:N2+1])
Z = zeros((M2,M2))
I = identity(M2)
L = kron(D1+dot(R,E1), identity(M)) + kron(D2+dot(R,E2),vstack((hstack([Z,I]),hstack([I,Z])))) + kron(matrix_power(R,2),D2t)

# Right-hand side and solution for u:
[rr,tt] = meshgrid(r[1:N2+1],t)
rr = hstack(stack(rr[:],axis=-1)); tt = hstack(stack(tt[:],axis=-1));
f = -rr**2*sin(tt/2.0)**4 + sin(6*tt)*cos(tt/2.0)**2
u = solve(L,f)

# Reshape results onto 2D grid and plot them:
u = reshape(u, (N2,M)).T
u2 = vstack((u[M-1,:],u[0:M-1,:]))
u = hstack([zeros((M,1)),u2])
[rr,tt] = meshgrid(r[0:N2+1],hstack([t[M-1],t[0:M-1]]))
xx = rr*cos(tt)
yy = rr*sin(tt)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, u, rstride=1, cstride=1, cmap="coolwarm", alpha=0.3)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-0.01, 0.05)
ax.set_zticks([-0.01, 0.05])
plt.show()