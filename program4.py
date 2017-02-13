from numpy import *
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

N = 24
h = 2*pi/N
x = linspace(0, 2*pi, N+1)[:-1]
col = hstack((0, 0.5*(-1.)**(arange(1, N))/tan(arange(1, N)*h/2.)))
D = toeplitz(col, -col)

v = maximum(0, 1-abs(x-pi)/2.)
plt.subplot(2, 2, 1)
plt.plot(x, v, '-bo', markersize=3, linewidth=1)
plt.axis([0, 2*pi, -0.5, 1.5])
plt.title('function')

plt.subplot(2, 2, 2)
plt.plot(x, D.dot(v), '-bo', markersize=3, linewidth=1)
plt.axis([0, 2*pi, -1, 1])
plt.title('spectral derivative')

plt.subplot(2, 2, 3)
v = exp(sin(x))
vprime = cos(x)*v
plt.plot(x, v, '-bo', markersize=3, linewidth=1)

plt.subplot(2, 2, 4)
plt.plot(x, D.dot(v), '-bo', markersize=3, linewidth=1)
plt.axis([0, 2*pi, -2, 2])
error = linalg.norm(D.dot(v)-vprime, inf)
plt.text(2.0, 1.4, 'max error = {0:2.2e}'.format(error))
plt.show()

