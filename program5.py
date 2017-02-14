# -*- coding: utf-8 -*-
"""
Spectral methods in MATLAB. Lloyd
Program 5
Created on Tue Feb  7 02:41:10 2017

@author: Anna
"""

# Repetition of Program 4 via FFT


from numpy import *
from scipy.linalg import norm, toeplitz
from matplotlib import pyplot as plt

#Differentiation of a hat function:
N = 24
h = 2*pi/N
x = arange(0,N)*h
v = maximum(0,1-abs(x-pi)/2.0)
v_hat = fft.fft(v)
w_hat = 1j*fft.fftfreq(N, 1./N)*v_hat#hstack((arange(0,N/2.0),arange(-N/2.0,-1))).T*(v_hat)
w = real(fft.ifft(w_hat))

plt.figure(1)
plt.subplot(221)
plt.plot(x, v, marker='o', linestyle='-')
plt.axis([-0, 2*pi, -0.5, 1.5])
plt.subplot(222)
plt.plot(x,w, marker='o', linestyle='-')
plt.axis([-0, 2*pi, -1, 1])

#Differentiation of exp(sin(x)):
v = exp(sin(x))
vprime = cos(x)*v
v_hat = fft.fft(v)
w_hat = 1j*fft.fftfreq(N, 1./N)*v_hat#hstack((arange(0,N/2.0),arange(-N/2.0,-1))).T*(v_hat)
w = real(fft.ifft(w_hat))
plt.subplot(223)
plt.plot(x, v, marker='o', linestyle='-')
plt.axis([-0, 2*pi, 0, 3])
plt.subplot(224)
plt.plot(x,w, marker='o', linestyle='-')
plt.axis([-0, 2*pi, -2, 2])
error = norm(w-vprime,inf)
plt.text(2, 1, 'max error = %.04e' %error)
plt.show()

