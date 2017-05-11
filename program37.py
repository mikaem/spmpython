"""
Spectral methods in MATLAB. Lloyd
Program 37
"""

# 2D "wave tank" with Neumann BCs for |y|=1


from numpy import *
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
from cheb import cheb
from numpy.linalg import matrix_power,solve
from scipy.linalg import toeplitz

# x variable in [-A,A], Fourier:
A = 3
Nx = 50
dx = 2.0*A/Nx
x = -A+dx*arange(1,Nx+1)
col = hstack((-1.0/(3*(dx/A)**2)-1.0/6, 0.5*(-1)**arange(2,Nx+1)/sin((pi*dx/A)*arange(1,Nx)/2.0)**2))
D2x = (pi/A)**2*toeplitz(col, col)

# y variable in [-1.1], Chebyshev:
Ny = 5
Dy,y = cheb(Ny)
D2y = matrix_power(Dy, 2)
rows = [[0,0],[Ny,Ny]]
columns = [[0,Ny],[0,Ny]]              
BC = solve(-Dy[rows,columns],Dy[[0,Ny],1:Ny])

# Grid and initial data:
[xx,yy] = meshgrid(x,y)
dt = 5.0/(Nx+Ny**2)
vv = exp(-8.0*((xx+1.5)**2+yy**2))
vvold = exp(-8.0*((xx+dt+1.5)**2+yy**2)) 


# Time-stepping by leap frog formula:                
plotgap = int(round(2.0/dt))
dt = 2.0/plotgap
fig = plt.figure()
for n in range(0,2*plotgap+1):
    t = n*dt
    if (((n+0.5)%plotgap) < 1):        #plots at multiplies of t = 1/3
        i = int(float(n)/plotgap) + 1
        ax = fig.add_subplot(3,1,i, projection='3d')
        ax.plot_surface(xx, yy, vv, rstride=1, cstride=1, cmap="coolwarm", alpha=0.3)
        ax.set_xlim(-A, A)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-0.15, 1)
        #ax.text2D(2.5, 1, 0.05, 't = '+str(t), transform=ax.transAxes)
    vvnew = 2*vv - vvold + (dt**2)*(vv.dot(D2x)+D2y.dot(vv))
    vvold = vv
    vv = vvnew
    vv[[0,Ny],:] = BC.dot(vv[1:Ny,:])
plt.show()