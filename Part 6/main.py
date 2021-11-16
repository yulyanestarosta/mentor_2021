import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
from matplotlib.colors import LogNorm

Z = np.loadtxt('image_data.txt')

xmin, xmax, nx = 0, 500, 500
ymin, ymax, ny = 0, 500, 500
x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y)

def gaussian(x, y, x0, y0, sigma_x, sigma_y, A):
    return A * np.exp( -((x-x0)/sigma_x)**2 - ((y-y0)/sigma_y)**2)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, cmap='plasma')
ax.set_zlim(0, np.max(Z)+2)
plt.show()

def _gaussian(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    arr.fill(args[-1])
    for i in range(len(args) // 5):
        arr += gaussian(x, y, *args[i*5:i*5+5])
    return arr


p0 = [92.61510202,  144.19705457, 59.90402952, 63.55289484,  11803.05219735,
      279.90243012,  293.00373364, 50, 50, 1000000, 1000]

xdata = np.vstack((X.ravel(), Y.ravel()))

popt, pcov = curve_fit(_gaussian, xdata, Z.ravel(), p0)
fit = np.zeros(Z.shape)

for i in range(len(popt)//5):
    fit += gaussian(X, Y, *popt[i*5:i*5+5])
print('Fitted parameters:')
print(popt)

rms = np.sqrt(np.mean((Z - fit)**2))
print('RMS residual =', rms)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, fit, cmap='plasma')
cset = ax.contourf(X, Y, Z-fit, zdir='z', cmap='plasma')
ax.set_zlim(0,np.max(fit))
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.imshow(Z, origin='lower', cmap='plasma',
          extent=(x.min(), x.max(), y.min(), y.max()))
ax.contour(X, Y, fit, 20, norm=LogNorm(), colors='w')
plt.show()
