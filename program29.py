"""
Spectral methods in MATLAB. Lloyd
Program 29
"""

# Solve Poisson equation on the unit disk (compare program 16 and 28)

from numpy import *
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
from cheb import cheb
import numpy.polynomial.chebyshev as n_cheb
from mpl_toolkits.mplot3d import axes3d

# Laplacian in polar coordinates:

N = 31
N2 = (N - 1) // 2
[D, r] = cheb(N)
D2 = D @ D
D1 = D2[1:N2 + 1, 1:N2 + 1]
D2 = D2[1:N2 + 1, N - 1:N2:-1]
E1 = D[1:N2 + 1, 1:N2 + 1]
E2 = D[1:N2 + 1, N - 1:N2:-1]

# \theta<TAB> = θ coordinate, ranging from 0 to 2π (M must be even):
M = 40
dθ = 2 * pi / M
θ = dθ * arange(1, M + 1)
M2 = M // 2
col = hstack([
    -pi**2 / (3 * dθ**2) - 1 / 6,
    0.5 * (-1)**arange(2, M + 1) / sin(dθ * arange(1, M) / 2)**2
])
D2θ = toeplitz(col, col)

# Laplacian in polar coordinates:
R = diag(1 / r[1:N2 + 1])
Z = zeros((M2, M2))
I = identity(M2)
L = kron(D1 + R @ E1, identity(M)) + kron(
    D2 + R @ E2, vstack((hstack([Z, I]), hstack([I, Z])))) + kron(R @ R, D2θ)

# Right-hand side and solution for u:
[rr, θθ] = meshgrid(r[1:N2 + 1], θ)
rr = hstack(stack(rr[:], axis=-1))
θθ = hstack(stack(θθ[:], axis=-1))
f = -rr**2 * sin(θθ / 2)**4 + sin(6 * θθ) * cos(θθ / 2)**2
u = linalg.solve(L, f)

# Reshape results onto 2D grid and plot them:
u = reshape(u, (N2, M)).T
u2 = vstack((u[M - 1, :], u[0:M - 1, :]))
u = hstack([zeros((M, 1)), u2])
[rr, θθ] = meshgrid(r[0:N2 + 1], hstack([θ[M - 1], θ[0:M - 1]]))
xx = rr * cos(θθ)
yy = rr * sin(θθ)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, u, rstride=1, cstride=1, cmap='coolwarm', alpha=.3)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-.01, .05)
ax.set_zticks([-.01, .05])
plt.show()
