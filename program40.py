"""
Spectral methods in MATLAB. Lloyd
Program 40
"""

# Eigenvalues of Orr-Sommerfeld operator  (compare pr.38)

from numpy import *
from numpy.linalg import matrix_power,solve
from matplotlib import pyplot as plt
from cheb import cheb
from scipy.linalg import eig


R = 5772.0
for N in range(40,120,20):
    # 2nd and 4th - order differantiation matrices:
    D,x = cheb(N)
    D2 = matrix_power(D,2)
    D2 = D2[1:N,1:N]
    S = diag(hstack([0, 1.0/(1-x[1:N]**2),0]))
    D4 = (diag(1-x**2).dot(matrix_power(D,4)) - 8*diag(x).dot(matrix_power(D,3)) - 12*matrix_power(D,2)).dot(S)
    D4 = D4[1:N,1:N]                 # boundary conditions
    
    # Orr-Sommerfeld operators A,B and generlized eigenvalues:
    I = identity(N-1)    
    A = (D4 - 2*D2 + I)/R - 2j*I - 1j*diag(1-x[1:N]**2).dot(D2-I)
    B = D2-I
    ee, V = eig(A,B)
    i = int(N/20.0-1)
    plt.subplot(2,2,i)
    plt.scatter(real(ee),imag(ee))
    plt.axis([-0.8, 0.2,-1,0],'square')
    
    plt.title('N = ' +str(N) + '\n lambda_max = '+str(max(real(ee))),fontsize=8)
    plt.grid('on')
plt.show()