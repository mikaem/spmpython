"""
Spectral methods in MATLAB. Lloyd
Program 39
"""

# Eigenmodes of biharmonic on a square with clamped BCs (compare pr.38)

from numpy import *
from numpy.linalg import matrix_power,solve
from matplotlib import pyplot as plt
from cheb import cheb
from scipy.linalg import eig
from scipy.interpolate import interp2d


# Construct spectral approximation to biharmonic operator:
N = 17
D,x = cheb(N)
D2 = matrix_power(D,2)
D2 = D2[1:N,1:N]
S = diag(hstack([0, 1.0/(1-x[1:N]**2),0]))
D4 = (diag(1-x**2).dot(matrix_power(D,4)) - 8*diag(x).dot(matrix_power(D,3)) - 12*matrix_power(D,2)).dot(S)
D4 = D4[1:N,1:N]                 # boundary conditions
I = identity(N-1)
L = kron(I,D4) + kron(D4,I) + 2*kron(D2,I).dot(kron(I,D2))

# Find and plot 25 eigenmodes:
Lam,V = eig(-L)
Lam = -real(Lam)
ii = argsort(Lam)
Lam = Lam[ii]
ii = ii[0:25]
V = real(V[:,ii])
Lam = sqrt(Lam/Lam[0])
xx,yy = meshgrid(x,x)
xxx,yyy = meshgrid(arange(-1,1.01,0.01),arange(-1,1.01,0.01))
sq = [1+1j,-1+1j,-1-1j,1-1j,1+1j]

for i in range(0,25):
    uu = zeros((N+1,N+1))
    uu[1:N,1:N] = (V[:,i]).reshape(N-1,N-1)
    plt.subplot(5,5,i+1)
    plt.hold('on')
    #plt.plot(sq)
    uuu = interp2d(xx, yy, uu, kind='cubic')
    plt.contour(xxx, yyy, uuu(arange(-1,1.01,0.01),arange(-1,1.01,0.01)),0)
    plt.axis([-1.25,1.25,-1.25,1.25], 'square')
    plt.axis('off')
    plt.title(str(Lam[i]),fontsize=7)

plt.show()