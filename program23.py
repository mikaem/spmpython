"""
Spectral methods in MATLAB. Lloyd
Program Program 23
"""

# Eigenvalues of perubated Laplacian on [-1,1]x[-1,1] (compare program 16)

from numpy import *
from scipy.linalg import eig
from matplotlib import pyplot as plt
from cheb import cheb
from scipy.interpolate import interp2d
from numpy.linalg import matrix_power, norm


# Set up tensor product Laplacian and compute 4 eigenmodes

N = 16
[D,x] = cheb(N)
y = x
[xx,yy] = meshgrid(x[1:N],y[1:N])
xx = hstack(xx[:]); yy = hstack(yy[:])
D2 = matrix_power(D, 2)
D2 = D2[1:N,1:N]
I = identity(N-1)
L = -kron(I,D2) - kron(D2,I)                  # Laplacian
L = L + diag(exp(20*(yy-xx-1)))               # + pertubation
D, V = eig(L)
ii = argsort(D)
D = D[ii]
ii = ii[0:4]
V = V[:,ii]

# Reshape them to 2D grid, interpolate to finer grid, and plot:

fine = arange(-1,1.02,0.02)
xxx,yyy = meshgrid(fine,fine)
uu = zeros((N+1,N+1))
for i in range(0,4):
    uu[1:N,1:N]=(V[:,i]).reshape(N-1,N-1)
    uu = uu/norm(uu[:],inf)
    uuu = interp2d(x, y, uu, kind='cubic')
    plt.subplot(2,2,i+1)
    plt.contour(xxx, yyy, uuu(fine,fine))
    plt.axis([-1,1,-1,1])
    plt.title('eig = '+str(D[i]/(pi**2/4))+'pi*2/4')
    
plt.show()
