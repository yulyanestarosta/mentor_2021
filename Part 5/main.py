import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import csv
import pandas as pd


x = np.empty(549)
y = np.empty(549)
z = np.empty(549)
errors = np.empty(549)


results = []
x= []; y=[]
with open('J0400_0550.csv') as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        results.append(row)
x, y, z = zip(*results)

plt.plot(x, y, 'o', markersize = 1.3)
plt.errorbar(x, y, xerr = 0, yerr = z, elinewidth= 0.3, linestyle = 'None' )


def absolute(x):
    if x <= 0:
        return -x
    else:
        return x

def flare(x, s, t_max, tau):
    return s * np.exp((-absolute(t_max - x) / tau))

Y = list(y)
sum = list(y)
for i in range(549):
    sum[i] = 0
for i in range(549):
    Y[i] = flare(x[i], 0.27, 55000, 300)
    sum[i] += Y[i]
plt.plot(x, Y, color = 'r', linewidth = 0.5)
for i in range(549):
    Y[i] = flare(x[i], 0.225, 55600, 300)
    sum[i] += Y[i]
plt.plot(x, Y, color = 'r', linewidth = 0.5)
for i in range(549):
    Y[i] = flare(x[i], 0.275, 56450, 300)
    sum[i] += Y[i]
plt.plot(x, Y, color = 'r', linewidth = 0.5)
for i in range(549):
    Y[i] = flare(x[i], 0.2, 57000, 300)
    sum[i] += Y[i]
plt.plot(x, Y, color = 'r', linewidth = 0.5)
for i in range(549):
    Y[i] = flare(x[i], 0.36, 58200, 1000)
    sum[i] += Y[i]
plt.plot(x, Y, color = 'r', linewidth = 0.3)
plt.plot(x, sum, color = 'b', linewidth = 0.5)
# plt.savefig("podgon.png", dpi = 900)
plt.show()



