import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.optimize as opt
import numpy as np
import csv
import pandas as pd

x = np.empty(549)
y = np.empty(549)
z = np.empty(549)
errors = np.empty(549)

results = []
x= []
y=[]
with open("J0400_0550.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        results.append(row)
x, y, z = zip(*results)

plt.plot(x, y, 'o', markersize = 1)
plt.errorbar(x, y, xerr = 0, yerr = z, elinewidth= 0.3, color='y', linestyle = 'None' )

def flare(x, s, t_max, tau):
    return s * np.exp(-np.abs( - x + t_max) / tau)

def summa(x, *param):
    return param[0] * np.exp(-np.abs( - x + param[1]) / param[2]) + \
           param[3] * np.exp(-np.abs( - x + param[4]) / param[5]) + \
           param[6] * np.exp(-np.abs( - x + param[7]) / param[8]) + \
           param[9] * np.exp(-np.abs( - x + param[10]) / param[11]) + \
           param[12] * np.exp(-np.abs( - x + param[13]) / param[14])


Y = list(y)
popt, pcov = opt.curve_fit(summa, x, Y, p0=[0.27, 55000, 300, 0.225, 55600, 300,0.275, 56450, 300, 0.2, 57000, 300,0.36, 58200, 1000])
for i in range(len(popt)):
    print(popt[i])

# sum = list(y)
#
# for i in range(549):
#     sum[i] = 0
#
# for i in range(549):
#     Y[i] = flare(x[i], 0.24612153479607002, 54939.85917734732, 523.2180640958908)
#     sum[i] += Y[i]
# plt.plot(x, Y, color = 'r', linewidth = 0.5)
#
# for i in range(549):
#     Y[i] = flare(x[i],0.14432587103528932, 55606.15297375751, 134.6701764507398)
#     sum[i] += Y[i]
# plt.plot(x, Y, color = 'r', linewidth = 0.5)
#
# for i in range(549):
#     Y[i] = flare(x[i],0.2301219823418467,
#     56445.90827055363,
#     459.4436355451555)
#     sum[i] += Y[i]
# plt.plot(x, Y, color = 'r', linewidth = 0.5)
#
# for i in range(549):
#     Y[i] = flare(x[i], 0.0997517671409234,
#     57016.754043021894,
#     159.2305471641201
#     )
#     sum[i] += Y[i]
# plt.plot(x, Y, color = 'r', linewidth = 0.5)
#
# for i in range(549):
#     Y[i] = flare(x[i], 0.3383415600845522,
#     58218.19316895285,
#     1517.0281131641518)
#     sum[i] += Y[i]
# plt.plot(x, Y, color = 'r', linewidth = 0.3)
#
# plt.plot(x, sum, color = 'b', linewidth = 0.5)
# plt.savefig("podgon1.png", dpi = 900)
plt.show()

# a = 1.548 * 10**(-32)
# dl = 1.31 * 10 ** 26
# nu = 15
# z = 0.761
#
# for i in range(0, 15, 3):
#
#     print(a * popt[i] * dl ** 2 / nu ** 2 / popt[i+2] ** 2 / (1 + z))
#
#     print((a * popt[i] * dl ** 2 / nu ** 2 / popt[i+2] ** 2 / (1 + z) / 5 / 10 ** 10)**(1/3))
