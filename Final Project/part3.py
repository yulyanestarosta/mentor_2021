import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([0, 0.0547945, 1.05205, 1.37808])
y1 = np.array([9.563832884146466, 11.525001747134848, 11.858011259168398, 13.642966062215299])
A1 = np.vstack([x1, np.ones(len(x1))]).T
m1, c1 = np.linalg.lstsq(A1, y1)[0]
print(m1, c1)
plt.errorbar(x1, y1, yerr=0.01622616086904319, color='k', linestyle='None')
plt.plot(x1, y1, 'o', color='k', markersize=2)
x = np.linspace(-0.01, 1.4, 100)
y = m1 * x + c1
plt.plot(x, y, color='b')

plt.minorticks_on()
plt.grid(which='major', color='gray', linewidth=0.5)

plt.title('Distance versus time dependence')
plt.xlabel('Time, years')
plt.ylabel('Distance, mas')

print((1 / (len(x1) - 1) * (np.var(y1) / np.var(x1) - m1 ** 2)) ** (1 / 2))
plt.savefig('pic3.png', dpi=900)
plt.show()

d = [2.9454460397664364,
11.319226471174266,
11.016965997147508,
22.050459496080535,
9.794929667066727,
15.633631363522158,
7.806456777862537,
10.666786928273375,
19.1436048538587,
10.10604858671961,
16.489758197540592,
20.56515485018235,
7.057356440635417]

b = 3.90419

for i in range(len(d)):
    print(np.arctan(2 * b / (b**2 + d[i]**2 - 1 )) * 180 / np.pi)

