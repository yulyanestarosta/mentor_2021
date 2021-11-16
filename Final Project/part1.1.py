import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib.ticker as ticker
from astropy.visualization import (imshow_norm, MinMaxInterval, SqrtStretch)
from matplotlib import ticker, cm

file = fits.open('0851_202_u_2001_06_30_icn.fits')
file.info()

data = file[0].data
data = data.squeeze()
data1 = data[210:290, 240:320]

maxi = 0
maxj = 0
max = data1[0][0]
for i in range(70):
    for j in range(80):
        if data1[i][j] >= max:
            max = data[i][j]
            maxi = i
            maxj = j
print(maxi, maxj)

# im, norm = imshow_norm(data1, ax, origin='lower', cmap='plasma')

fig, ax = plt.subplots()
cs = ax.contour(data1, levels=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 2e0], cmap='plasma')

# x_label_list = ['-20', '-10', '0', '10', '20']
# ax.set_xticks([0, 10, 20, 30, 40])
# ax.set_xticklabels(x_label_list)

def diagram(length, size=0.1): ## size - размер пикселя, length - длина изображения по вертикали

    hdulist1 = fits.open('0851_202_u_2001_06_30_icn.fits')
    hdu1 = hdulist1[0]
    dlen = length
    a = hdu1.header["BMAJ"] * 3600000 / size / 2
    alpha = hdu1.header["BPA"]
    b = hdu1.header["BMIN"] * 3600000 / size / 2
    scale = 1
    x = 1.5 * a
    x1 = 1.5 * b
    line1 = plt.Line2D((x1 + scale * a * np.sin(alpha * np.pi / 180),
                        x1 - scale * a * np.sin(alpha * np.pi / 180)),
                       (dlen - x + scale * a * np.cos(alpha * np.pi / 180),
                        dlen - x - scale * a * np.cos(alpha * np.pi / 180)),
                       lw=1,
                       color="black")
    plt.gca().add_line(line1)
    line2 = plt.Line2D((x1 + scale * b * np.cos(alpha * np.pi / 180),
                        x1 - scale * b * np.cos(alpha * np.pi / 180)),
                       (dlen - x - scale * b * np.sin(alpha * np.pi / 180),
                        dlen - x + scale * b * np.sin(alpha * np.pi / 180)),
                       lw=1,
                       color="black")
    plt.gca().add_line(line2)
diagram(80)

x_label_list = ['0', '-2', '-4', '-6']
ax.set_xticks([16, 36, 56, 76])
ax.set_xticklabels(x_label_list)

y_label_list = ['-4', '-2', '0', '2']
ax.set_yticks([8, 28, 48, 68])
ax.set_yticklabels(y_label_list)

plt.xlabel('Relative right ascention (mas)')
plt.ylabel('Relative declination (mas)')
plt.title('0851+202, 2001.30.06')
plt.savefig('pic11.png', dpi=900)
plt.show()

plt.show()


