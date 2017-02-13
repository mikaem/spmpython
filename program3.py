from numpy import *
import matplotlib.pyplot as plt

h = 1.
xmax = 10.
N = int(2*xmax/h+1)
x = linspace(-xmax, xmax, N)
xx = linspace(-xmax-h/20., xmax+h/20, int(20*xmax/h+1))

for i in range(1, 4):
    if i == 1:
        v = (x==0)

    elif i == 2:
        v = abs(x) <= 3

    elif i == 3:
        v = maximum(0, 1-abs(x)/3.)

    plt.subplot(3, 1, i)
    plt.plot(x, v, 'bo')
    p = zeros(len(xx))
    for j, xj in enumerate(x):
        p += v[j]*sin(pi*(xx-xj)/h)/(pi*(xx-xj)/h)

    plt.plot(xx, p)
    plt.axis([-xmax, xmax, -0.5, 1.5])

plt.show()
