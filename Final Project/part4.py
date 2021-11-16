import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.optimize as opt
import numpy as np
import csv
import pandas as pd

x = np.empty(422)
y = np.empty(422)
z = np.empty(422)
errors = np.empty(422)

results = []
x= []
y=[]
with open("J0854_2006.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        results.append(row)
x, y, z = zip(*results)

plt.plot(x, y, 'o', markersize = 1)
plt.errorbar(x, y, xerr = 0, yerr = z, elinewidth= 0.3, color='g', linestyle = 'None' )

def flare(x, amp, t_max, tau):
    return amp * np.exp(-np.abs( - x + t_max) / tau)

def summa(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amp = params[i]
        tau = params[i + 2]
        t_max = params[i + 1]
        y += flare(x, amp, t_max, tau)
    return y

p0=[
    9, 54600, 140,
    9, 54900, 135,
    26.5, 55200, 130,
    14, 55565, 70,
    19, 55700, 110,
    6, 55910, 70,
    14, 56000, 120,
    10, 56210, 130,
    11, 56380, 45,
    9, 56520, 100,
    7, 56630, 80,
    12, 56850, 90,
    12, 56980, 80]
Y = list(y)
popt, pcov = opt.curve_fit(summa, x, Y, p0, maxfev=100000, bounds=(0,np.inf))
print(popt)

sum = list(y)
for i in range(422):
    sum[i] = 0
for j in range(len(p0)//3):
    for i in range(422):
        sum[i]+=flare(x[i], p0[j*3], p0[j*3+1], p0[j*3+2])/2.5

plt.plot(x, sum, color = 'b', linewidth = 0.5)
plt.xlabel('Date')
plt.ylabel('Flux density (Jy)')
plt.savefig('pic4.png', dpi=900)


plt.show()

a = 1.548 * 10**(-32)
dl = 4.943255e+25
nu = 15
z = 0.306

for i in range(0, len(popt), 3):

    print(a * popt[i] * dl ** 2 / nu ** 2 / popt[i+2] ** 2 / (1 + z))

    print((a * popt[i] * dl ** 2 / nu ** 2 / popt[i+2] ** 2 / (1 + z) / 5 / 10 ** 10)**(1/3))
