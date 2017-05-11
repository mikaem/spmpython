"""
Spectral methods in MATLAB. Lloyd
Program 34
"""

# Allen-Cahn eq. u_t = u_xx + u -u**3, u(-1)=-1, u(1)=1 (compare pr.6 and 32)


from numpy import *
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
from cheb import cheb
from numpy.polynomial import chebyshev as n_cheb
from numpy.linalg import matrix_power


#Differentiation matrix and initial data:
N = 20
D,x = cheb(N)
D2 = matrix_power(D, 2)                # use full size matrix
D2[[0,N],:] = zeros((2,N+1))                 # for convinience
eps = 0.01
dt = min([0.01, 50*N**(-4)/eps])
t = 0
v = 0.53*x + 0.47*sin(-1.5*pi*x)

# Solve PDE by Euler formula and plot results:
tmax = 100
tplot = 2
nplots = int(round(tmax/tplot))
plotgap = int(round(tplot/dt))
dt = tplot/plotgap
xx = arange(-1,1.025,0.025)
vh = n_cheb.chebfit(x, v, N)
vv = n_cheb.chebval(xx, vh)
plotdata = vstack((vv, zeros((nplots,len(xx)))))
tdata = t
for i in range(0,nplots):
    for n in range(0,plotgap):
        t += dt
        v = v +dt*(eps*D2.dot(v-x) + v - v**3)           #Euler
    vh = n_cheb.chebfit(x, v, N)
    vv = n_cheb.chebval(xx, vh)
    plotdata[i+1,:] = vv
    tdata = vstack((tdata, t))

fig = plt.figure()
ax = axes3d.Axes3D(fig)
X, Y = meshgrid(xx, tdata)
ax.plot_wireframe(X,Y,plotdata)
ax.set_xlim(-1, 1)
ax.set_ylim(0, tmax)
ax.set_zlim(-1, 1)
plt.show()
