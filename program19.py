"""
Spectral methods in MATLAB. Lloyd
Program 19
"""

# 2nd order wave eq. on Chebyshev grid (compare program 6)

from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from chebfft import chebfft

N = 80
x = cos(pi*arange(0,N+1)/N)
dt = 8.0/N**2
v = exp(-200*x**2)
vold = exp(-200*(x-dt)**2)
tmax = 4
tplot = 0.075
plotgap = int(round(tplot/dt))
dt = tplot/plotgap
nplots = int(round(tmax/tplot))
plotdata = vstack((v, zeros((nplots,N+1))))
tdata = 0
for i in range(0,nplots):
    for n in range(0,plotgap):
        w = chebfft(chebfft(v)).T
        w[0] = 0
        w[N] = 0
        vnew = 2*v - vold +dt**2*w
        vold = v
        v = vnew
    plotdata[i+1,:] = v
    tdata = vstack((tdata, dt*i*plotgap))

# Plot results

fig = plt.figure()
ax = axes3d.Axes3D(fig)
X, Y = meshgrid(x, tdata)
ax.plot_wireframe(X,Y,plotdata)
ax.set_xlim(-1, 1)
ax.set_ylim(0, tmax)
ax.set_zlim(-2, 2)
plt.show()