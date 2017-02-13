from numpy import *
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

Nvec = arange(2, 100, 2)
plt.figure()
errors = []
for N in Nvec:
    h = 2*pi/N
    x = linspace(-pi, pi, N+1)[:-1]
    u = exp(sin(x))
    uprime = cos(x)*u
    col = hstack((0, 0.5*(-1.)**(arange(1, N))/tan(arange(1, N)*h/2.)))
    D = toeplitz(col, -col)
    error = linalg.norm(D.dot(u)-uprime, inf)
    errors.append(error)

plt.loglog(Nvec, errors, 'bs')
plt.title('Convergence of spectral differentiation')
plt.xlabel('N')
plt.ylabel('error')
plt.show()
