from numpy import *
import matplotlib.pyplot as plt
from scipy.sparse import diags

Nvec = arange(2,100,2)
plt.figure()
errors = []
for N in Nvec:
    h = 2*pi/N
    x = linspace(-pi, pi, N+1)[:-1]
    u = exp(sin(x))
    uprime = cos(x)*u
    d = 0.5/tan(arange(1, N)*h/2.)
    ds = [(-1.)**((i-1)%2)*d[i-1]*ones(N-i) for i in range(1, N)]
    D = diags(ds, range(1, N))
    D = (D-D.T)
    error = linalg.norm(D.dot(u)-uprime, inf)
    errors.append(error)

plt.loglog(Nvec, errors, 'bs')
plt.title('Convergence of spectral differentiation')
plt.xlabel('N')
plt.ylabel('error')
plt.show()


