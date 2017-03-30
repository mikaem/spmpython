"""
Spectral methods in MATLAB. Lloyd
Program 27
"""

# Solve KdV eq.: u_t + uu_x + u_xxx = 0 on [-pi,pi] by
# FFT with integrating factor v = exp(-ik**3t)*u-hat

from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d


# Set up grid and two-solution initial data:
N = 256
dt = 0.4/N**2
x = (2*pi/N)*arange(-N/2.0,N/2.0)
A = 25
B = 16
u = 3*A**2*(1.0/cosh(0.5*(A*(x+2))))**2 + 3*B**2*(1.0/cosh(0.5*(B*(x+1))))**2
v = fft.fft(u)
k = hstack([arange(0,N/2.0),0,arange(-N/2.0+1,0)])
ik3 = 1j*k**3


# Solve PDE and plot results:
tmax = 0.006
nplt = floor((tmax/25)/dt)
nmax = int(round(tmax/dt))
udata = [u]
tdata = [0]
for n in range(1,nmax+1):
    t = n*dt
    g = -0.5j*dt*k
    E = exp(dt*ik3/2)
    E2 = E**2
    a = g*fft.fft(real(fft.ifft(v))**2)
    b = g*fft.fft(real(fft.ifft(E*(v + a/2.0)))**2)          # 4th-order
    c = g*fft.fft(real(fft.ifft(E*v + b/2.0))**2)            # Runge-Kutta
    d = g*fft.fft(real(fft.ifft(E2*v + E*c))**2)
    v = E2*v + (E2*a + 2*E*(b+c) +d)/6.0
    if ((n%int(nplt))==0):
        u = real(fft.ifft(v))
        udata.append(u)
        tdata.append(t)

fig = plt.figure()
ax = axes3d.Axes3D(fig)
X, Y = meshgrid(x, tdata)
ax.plot_wireframe(X,Y,udata)
ax.set_xlim(-pi, pi)
ax.set_ylim(0, tmax)
ax.set_zlim(0, 2000)
ax.set_xlabel('x')
ax.set_ylabel('t')
plt.show()