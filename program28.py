"""
Spectral methods in MATLAB. Lloyd
Program 28
"""

# Eigenmodes of Laplacian on the disk (compare program 22)

from numpy import *
from scipy.special import airy
from scipy.linalg import toeplitz
from matplotlib import pyplot as plt
from cheb import cheb
from numpy.polynomial import chebyshev as n_cheb
from numpy.linalg import eig,matrix_power, norm
from mpl_toolkits.mplot3d import axes3d

# r coordinate, ranging from -1 to 1 (N must be odd)

N = 25
N2 = int((N-1)/2.0)
[D,r] = cheb(N)
D2 = matrix_power(D, 2)
D1 = D2[1:N2+1,1:N2+1]
D2 = D2[1:N2+1,N-1:N2:-1]
E1 = D[1:N2+1,1:N2+1]
E2 = D[1:N2+1,N-1:N2:-1]

# t = theta coordinate, ranging from 0 to 2*pi (M must be even):
M = 20
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

# Compute four eigenmodes:
index = [0,2,5,8]
Lam, V = eig(-L)
ii = argsort(Lam)
Lam = Lam[ii]
#V = V[:,ii]
ii = ii[index]
V = real(V[:,ii])
Lam = sqrt(Lam[index]/Lam[0])

# Plot eigenmodes with nodal lines underneath:
[rr,tt] = meshgrid(r[0:N2+1],hstack([0,t]))
xx = rr*cos(tt)
yy = rr*sin(tt)
z = exp(1j*pi*arange(-100,101)/100)
fig = plt.figure()
for i in range(0,4):
    u = reshape(V[:,i], (N2,M)).T
    u = hstack([zeros((M+1,1)),vstack([u[M-1,:],u[0:M,:]])])
    u = u/norm(u)
    ax = fig.add_subplot(2,2,i+1, projection='3d')
    plt.hold('on')
    ax.plot_surface(xx, yy, u, rstride=1, cstride=1, cmap="coolwarm", alpha=0.3)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_axis_off()
    #plt.contour(xx, yy, u-1)
    #plt.plot()

plt.show()