import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

xfile = open('x_data.txt')
x_arr = np.empty(1000)
i = 0
for x in xfile:
    x_arr[i] = x
    i += 1

yfile = open('y_data_nonlinear.txt')
y_arr = np.empty(1000)
i = 0
for y in yfile:
    y_arr[i] = y
    i += 1


def func(x, a, b, c):
    return a * np.exp(-1 * b * np.sin(x)) + c * x


popt, pcov= opt.curve_fit(func, x_arr, y_arr)
plt.plot(x_arr, y_arr, 'o', color='k', markersize=0.5)
plt.plot(x_arr, func(x_arr, popt[0], popt[1], popt[2]))
print(popt)
plt.show()
