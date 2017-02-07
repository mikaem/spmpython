from numpy import *
import matplotlib.pyplot as plt
from scipy.sparse import diags

Nvec = 2**(arange(3,12))
plt.figure()
errors = []
for N in Nvec:
    h = 2*pi/N
    x = linspace(-pi, pi, N+1)[:-1]
    u = exp(sin(x))
    uprime = cos(x)*u
    D = diags([2./3.*ones(N-1), -1./12.*ones(N-2), 1./12.*ones(2), -2./3.], [1,2,N-2,N-1])
    D = (D-D.T)/h
    error = linalg.norm(D.dot(u)-uprime, inf)
    errors.append(error)

plt.loglog(Nvec, errors, 'c')
plt.title('Convergence of 4th-order finite differences')
plt.semilogy(Nvec, (1.*Nvec)**(-4), '--')
plt.text(105, 5e-8, r'$N^{-4}$')
plt.xlabel('N')
plt.ylabel('error')
plt.show()


