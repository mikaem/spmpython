"""
Spectral methods in MATLAB. Lloyd
Program 25
"""

# Stability regions for ODE formulas

from numpy import *
from matplotlib import pyplot as plt

# Adams-Bashforth:
plt.subplot(2,2,1)
plt.hold('on')
plt.plot([-8,8],[0,0],[0,0],[-8,8])
z = exp(1j*pi*arange(0,201)/100)
r = z-1
s = 1.0                                  # order 1
plt.plot(real(r/s),imag(r/s))
s = (3-1.0/z)/2.0                        # order 2
plt.plot(real(r/s),imag(r/s))
s = (23-16.0/z+5.0/z**2)/12.0
plt.plot(real(r/s),imag(r/s))            # order 3
plt.axis([-2.5,0.5,-1.5,1.5],'equal')
plt.grid('on')
plt.title('Adams-Bashforth')

# Adams-Moulton:
plt.subplot(2,2,2)
plt.hold('on')
plt.plot([-8,8],[0,0],[0,0],[-8,8])
s = (5*z+8-1.0/z)/12.0                   # order 3
plt.plot(real(r/s),imag(r/s))
s = (9*z+19-5.0/z+1.0/z**2)/24.0         # order 4
plt.plot(real(r/s),imag(r/s))
s = (251*z+646-264.0/z+106.0/z**2-19.0/z**3)/720.0     # order 5
plt.plot(real(r/s),imag(r/s))
d = 1 - 1.0/z
s = 1 - d/2.0-d**2/12.0-d**3/24.0-19*d**4/720.0-3*d**5/160.0    # order 6
plt.plot(real(d/s),imag(d/s))
plt.axis([-7,1,-4,4],'equal')
plt.grid('on')
plt.title('Adams-Moulton')

# Backward differentiation:
plt.subplot(2,2,3)
plt.hold('on')
plt.plot([-40,40],[0,0],[0,0],[-40,40])
r = 0
for i in range(1,7):
    r += (d**i)/i                    # orders 1-6
    plt.plot(real(r),imag(r))            
plt.axis([-15,35,-25,25],'equal')
plt.grid('on')
plt.title('Backward differentiation')


# Runge-Kutta:
plt.subplot(2,2,4)
plt.hold('on')
plt.plot([-8,8],[0,0],[0,0],[-8,8])
w = 0
W = []
W.append(w)
for i in range(1,len(z)):              # order 1
    w -= (1+w-z[i])
    W.append(w)
plt.plot(real(W),imag(W))
w = 0
W = []
W.append(w)
for i in range(1,len(z)):              # order 2
    w -= (1+w+0.5*w**2-z[i]**2)/(1+w)
    W.append(w)
plt.plot(real(W),imag(W))  
w = 0
W = []
W.append(w)
for i in range(1,len(z)):              # order 3
    w -= (1+w+0.5*w**2+w**3/6.0-z[i]**3)/(1+w+w**2/2)
    W.append(w)
plt.plot(real(W),imag(W))
w = 0
W = []
W.append(w)
for i in range(1,len(z)):              # order 4
    w -= (1+w+0.5*w**2+w**3/6.0+w**4/24.0-z[i]**4)/(1+w+w**2/2+w**3/6)
    W.append(w)
plt.plot(real(W),imag(W))
plt.axis([-5,2,-3.5,3.5],'equal')
plt.grid('on')
plt.title('Runge-Kutta')   

plt.show()