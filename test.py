import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib.ticker as ticker
from astropy.visualization import (imshow_norm, MinMaxInterval, SqrtStretch)

M42 = fits.open('1226+023.u.2009_12_05.icn.fits')
M42.info()

data = M42[0].data
data = data.squeeze()
data1 = data[900:1110, 1000:1210]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x_label_list = ['-20', '-10', '0', '10', '20']
ax.set_xticks([0, 50, 100, 150, 200])
ax.set_xticklabels(x_label_list)
im, norm = imshow_norm(data1, ax, origin='lower', cmap='plasma')

# plt.contour(data1)


plt.xlabel('Relative right ascension (mas)')
fig.colorbar(im)
plt.show()


# plt.imshow(data.squeeze(), cmap='gist_rainbow', norm=LogNorm())
# cbar = plt.colorbar(ticks=[5.e1, 1.e2, 2.e4])
#
# cbar.ax.set_yticklabels(['5,000', '10,000', '20,000'])
# plt.show()
