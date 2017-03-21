"""
Spectral methods in MATLAB. Lloyd
Program 20
"""

# 2nd order wave eq. in 2D via FFT (compare program 19)

from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import interp2d
from numpy.linalg import matrix_power

# Grid and initital data
N = 24
x = cos(pi*arange(0,N+1)/N)
y = x.T
dt = 6.0/N**2
[xx,yy] = meshgrid(x,y)
plotgap = int(round((1./3)/dt))
dt = (1./3)/plotgap
vv = exp(-40*((xx-0.4)**2+yy**2))
vvold = vv

# Time-stepping by leap frog formula:
fig = plt.figure()
for n in range(0,3*plotgap+1):
    t = n*dt
    if (((n+0.5)%plotgap) < 1):        #plots at multiplies of t = 1/3
        i = int(float(n)/plotgap) + 1
        ar = arange(-1,1+1./16,1./16)
        [xxx,yyy] = meshgrid(ar,ar)
        vvv = interp2d(x, y, vv, kind='cubic')
        ax = fig.add_subplot(2,2,i, projection='3d')
        ax.plot_surface(xxx, yyy, vvv(ar,ar), rstride=1, cstride=1, cmap="coolwarm", alpha=0.3)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-0.15, 1)
        ax.text2D(0.05, 0.95, 't = '+str(round(t, 2)), transform=ax.transAxes)
    uxx = zeros((N+1,N+1))
    uyy = zeros((N+1,N+1))
    ii = arange(1,N)
    for i in range(1,N+1):          # 2nd derivs wrt x in each row
        v = vv[i,:]
        V = hstack([v,flipud(v[ii])])
        U = real(fft.fft(V))
        W1 = real(fft.ifft(1j*hstack([arange(0,N), 0, arange(1-N,0)]).T*U))     # diff wrt theta
        W2 = real(fft.ifft(-hstack([arange(0,N+1), arange(1-N,0)]).T**2*U))       # diff**2 wrt theta
        uxx[i,ii] = W2[ii]/(1-x[ii]**2) - x[ii]*W1[ii]/(1-x[ii]**2)**(3./2)
    for k in range(1,N+1):
        v = vv[:,k]
        V = hstack([v,flipud(v[ii])])
        U = real(fft.fft(V))
        W1 = real(fft.ifft(1j*hstack([arange(0,N), 0, arange(1-N,0)]).T*U))     # diff wrt theta
        W2 = real(fft.ifft(-hstack([arange(0,N+1), arange(1-N,0)]).T**2*U))       # diff**2 wrt theta
        uyy[ii,k] = W2[ii]/(1-y[ii]**2) - y[ii]*W1[ii]/(1-y[ii]**2)**(3./2)
    vvnew = 2*vv - vvold + (uxx+uyy)*dt**2
    vvold = vv
    vv = vvnew
plt.show()
