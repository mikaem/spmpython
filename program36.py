"""
Spectral methods in MATLAB. Lloyd
Program 36
"""

# Laplace equation on [-1,1]x[-1,1] with nonzero boundary conditions

from numpy import *
from cheb import cheb
from numpy.linalg import matrix_power,solve
from matplotlib import pyplot as plt
from scipy.interpolate import interp2d
from mpl_toolkits.mplot3d import axes3d
from pylab import find

# Set up grid and 2D Laplacian, boundary points included:

N = 24
D,x = cheb(N)
y = x
xx,yy = meshgrid(x,y)
xx = hstack(xx[:]); yy = hstack(yy[:]);                   # strech 2D grids to 1D vectors
D2 = matrix_power(D, 2)
I = identity(N+1)
L = kron(I,D2) + kron(D2,I)          # Laplacian

# Impose boundary conditions by replacing appropriate rows of L:
b = find(logical_or((abs(xx)==1),(abs(yy)==1)))
L[b,:] = zeros((4*N,(N+1)**2))
L[[b],[b]] = 1#eye(4*N)
rhs = zeros((N+1)**2)
rhs[b] = (yy[b]==1)*(xx[b]<0)*sin(pi*xx[b])**4+0.2*(xx[b]==1)*sin(3*pi*yy[b])


# Solve Laplace equation, reshape long 1D results onto 2D grid, and plot:
u = solve(L,rhs)
uu=u.reshape(N+1,N+1)
xx,yy = meshgrid(x,y)
ar = arange(-1,1.04,0.04)
xxx,yyy = meshgrid(ar,ar)
uuu = interp2d(x, y, uu, kind='cubic')

fig = plt.figure()
ax = axes3d.Axes3D(fig)
ax.plot_surface(xxx, yyy, uuu(ar,ar), rstride=1, cstride=1, cmap="coolwarm", alpha=0.3)
ax.text(0, 0.8, 0.4, "u(0,0)="+str(uu[int(N/2.0),int(N/2.0)]))
plt.show()