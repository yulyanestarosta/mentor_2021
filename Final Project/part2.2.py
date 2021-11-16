import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
from matplotlib.colors import LogNorm
from astropy.io import fits

file = fits.open('0851_202_u_2002_07_19_icn.fits')
file.info()

data = file[0].data
data = data.squeeze()
Z = data[210:300, 235:325]

xmin, xmax, nx = 0, 90, 90
ymin, ymax, ny = 0, 90, 90
x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y)

def gaussian(x, y, x0, y0, sigma_x, sigma_y, A):
    return A * np.exp( -((x-x0)/sigma_x)**2 - ((y-y0)/sigma_y)**2)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, Z, cmap='plasma')
# ax.set_zlim(0, np.max(Z)+2)
# plt.show()

def _gaussian(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    arr.fill(args[-1])
    for i in range(len(args) // 5):
        arr += gaussian(x, y, *args[i*5:i*5+5])
    return arr


p0 = [20, 47, 1, 1, 5, 30, 50, 1, 1, 3]

xdata = np.vstack((X.ravel(), Y.ravel()))

popt, pcov = curve_fit(_gaussian, xdata, Z.ravel(), p0, maxfev=10000)
fit = np.zeros(Z.shape)

for i in range(len(popt)//5):
    fit += gaussian(X, Y, *popt[i*5:i*5+5])
print('Fitted parameters:')
print(popt)

rms = np.sqrt(np.mean((Z - fit)**2))
print('RMS residual =', rms)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, fit, cmap='plasma')
# cset = ax.contourf(X, Y, Z-fit, zdir='z', cmap='plasma')
# ax.set_zlim(0,np.max(fit))
# plt.show()

fig = plt.figure()
ax = fig.add_subplot()
# ax.imshow(Z, origin='lower', cmap='plasma',
#           # extent=(x.min(), x.max(), y.min(), y.max())
#           )

x_label_list = ['2', '0', '-2', '-4', '-6']
ax.set_xticks([0, 20, 40, 60, 80])
ax.set_xticklabels(x_label_list)

y_label_list = ['-4', '-2', '0', '2', '4']
ax.set_yticks([7, 27, 47, 67, 87])
ax.set_yticklabels(y_label_list)
plt.xlabel('Relative right ascention (mas)')
plt.ylabel('Relative declination (mas)')
plt.title('0851+202, 2002.19.07')
ax.contour(X, Y, fit, levels=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 2e0], cmap='plasma')
plt.savefig('pic22.png', dpi=900)
plt.show()
print(popt[5] - popt[0])
print(popt[6] - popt[1])

print(((popt[5] - popt[0])**2 + (popt[6] - popt[1])**2)**0.5)