import numpy as np
from matplotlib import pyplot as plt
import shenfun
from shenfun.operators import div, grad, Dx
from shenfun import inner_product
from shenfun.fourier.bases import R2CBasis
from mpl_toolkits.mplot3d import axes3d
from matplotlib.collections import PolyCollection
from time import time

N = 256
t = 0
dt = 0.1/N**2
tstep = 0
tmax = 0.006
tsteps = int(tmax/dt)
nplt = int(tmax/25/dt)
SD = R2CBasis(N, True)
Nh = SD.spectral_shape()
x = SD.points_and_weights()[0]
u = np.zeros(N)
u_1 = np.zeros(N)
u_2 = np.zeros(N)
u_ab = np.zeros(N)

u_hat = np.zeros(Nh, np.complex)
u_hat_1 = np.zeros(Nh, np.complex)
u_hat_2 = np.zeros(Nh, np.complex)
uu_hat = np.zeros(Nh, np.complex)

k = SD.wavenumbers(N)
A = 25.
B = 16.

# initialize
u[:] = 3*A**2/np.cosh(0.5*A*(x-np.pi+2))**2 + 3*B**2/np.cosh(0.5*B*(x-np.pi+1))**2

u_1[:] = u
u_2[:] = u
d = 1.-2*dt*1j*k**3/3.
d1 = 1.-dt*1j*k**3

plt.figure()
u_hat = SD.forward(u, u_hat)
u_hat_1[:] = u_hat
u_hat_2[:] = u_hat
data = []
tdata = []

#t0 = time()
#for i in range(tsteps):
    #t += dt
    #tstep += 1
    #u_ab[:] = 2*u_1 - u_2
    #uu_hat = SD.forward(u_ab*u_ab, uu_hat)
    #if tstep == 1:
        #u_hat[:] = (u_hat_1 - dt*1j*k/2.*uu_hat)/d1
    #else:
        #u_hat[:] = (4./3.*u_hat_1 - 1./3.*u_hat_2 - dt*1j*k/3.*uu_hat)/d

    #u_hat_2[:] = u_hat_1
    #u_hat_1[:] = u_hat

    #u = SD.backward(u_hat, u)
    #u_2[:] = u_1
    #u_1[:] = u
    #if tstep % nplt == 0:
        #plt.plot(x, u)
        #plt.draw()
        #plt.pause(1e-6)
        #data.append(u.copy())
#print('Time={}'.format(time()-t0))

v = SD.test_function()
A = 3*inner_product(v, v) + 2*dt*inner_product(v, Dx(v, 0, 3))
B = inner_product(v, v)
C = inner_product(v, grad(v))
uu_hat0 = uu_hat.copy()
rhs = np.zeros(Nh, np.complex)
t0 = time()
for i in range(tsteps):
    t += dt
    tstep += 1
    u_ab[:] = 2*u_1 - u_2
    uu_hat = SD.forward(u_ab*u_ab, uu_hat)
    uu_hat0 = C.matvec(uu_hat, uu_hat0)
    u_hat = B.matvec(4*u_hat_1 - u_hat_2, u_hat)
    u_hat -= dt*uu_hat0

    if tstep == 1:
        u_hat[:] = (u_hat_1 - dt*1j*k/2.*uu_hat)/d1
    else:
        u_hat = A.solve(u_hat)

    u_hat_2[:] = u_hat_1
    u_hat_1[:] = u_hat

    u = SD.backward(u_hat, u)
    u_2[:] = u_1
    u_1[:] = u
    if tstep % nplt == 0:
        plt.plot(x, u)
        plt.draw()
        plt.pause(1e-6)
        data.append(u.copy())
print('Time={}'.format(time()-t0))


s = []
for d in data:
    s.append(np.vstack((x, d)).T)

N = len(data)
tdata = np.linspace(0, t, N)
ddata = np.array(data)


fig = plt.figure(figsize=(8,3))
ax = axes3d.Axes3D(fig)
X, Y = np.meshgrid(x, tdata)
ax.plot_wireframe(X, Y, ddata, cstride=1000)
ax.set_xlim(0, 2*np.pi)
ax.set_ylim(0, t)
ax.set_zlim(0, 2000)
ax.view_init(65, -105)
ax.set_zticks([0, 2000])
ax.grid()


fig2 = plt.figure(figsize=(8,3))
ax2 = fig2.gca(projection='3d')
poly = PolyCollection(s, facecolors=(1,1,1,1), edgecolors='b')
ax2.add_collection3d(poly, zs=tdata, zdir='y')
ax2.set_xlim3d(0, 2*np.pi)
ax2.set_ylim3d(0, t)
ax2.set_zlim3d(0, 2000)
ax2.view_init(65, -105)
ax2.set_zticks([0, 2000])
ax2.grid()

fig3 = plt.figure(figsize=(8,3))
ax3 = fig3.gca(projection='3d')
X, Y = np.meshgrid(x, tdata)
ax3.plot_surface(X, Y, ddata, cstride=1000, rstride=1, color='w')
ax3.set_xlim(0, 2*np.pi)
ax3.set_ylim(0, t)
ax3.set_zlim(0, 2000)
ax3.view_init(65, -105)
ax3.set_zticks([0, 2000])
ax3.grid()

fig4 = plt.figure(figsize=(8,3))
ax4 = fig4.gca(projection='3d')
for i in range(len(tdata)):
    ax4.plot(x, ddata[i], tdata[i])
ax4.view_init(65, -105)
ax4.set_zticks([0, 2000])
ax4.grid()

fig5 = plt.figure(facecolor='k')
ax5 = fig5.add_subplot(111, axisbg='k')
N = len(tdata)
for i in range(N):
    offset = (N-i-1)*200
    ax5.plot(x, ddata[N-i-1]+offset, 'w', lw=2, zorder=(i+1)*2)
    ax5.fill_between(x, ddata[N-i-1]+offset, offset, facecolor='k', lw=0, zorder=(i+1)*2-1)
plt.show()

