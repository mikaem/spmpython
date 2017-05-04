"""
Spectral methods in MATLAB. Lloyd
Program 31
"""

# Gamma function via complex integral, trapezoid rule

from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

N = 70
theta = -pi + (2*pi/N)*arange(0.5,N)
c = -11
r = 16
x = arange(-3.5,4.1,0.1)
y = arange(-2.5,2.6,0.1)
[xx,yy] = meshgrid(x,y)
zz = xx +1j*yy
gaminv = 0*zz
for i in range(0,N):
    t = c + r*exp(1j*theta[i])
    gaminv = gaminv + exp(t)*t**(-zz)*(t-c)
    
gaminv = gaminv/N
gam = 1./gaminv

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, abs(gam), rstride=1, cstride=1, cmap="coolwarm", alpha=0.3)
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.set_xlim(-3.5, 4)
ax.set_ylim(-2.5, 2.5)
ax.set_zlim(0, 6)
plt.show()