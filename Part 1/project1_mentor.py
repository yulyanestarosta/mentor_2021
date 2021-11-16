import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
import numpy as np

plt.style.use(astropy_mpl_style)

image_file = get_pkg_data_filename('1226+023.u.2009_12_05.icn.fits')
hdu_list = fits.open(image_file)
print(hdu_list[0].header)
image_data = hdu_list[0].data
print(type(image_data))

print(image_data.squeeze().shape)
x = np.linspace(10, -10)
plt.figure()
plt.imshow(image_data.squeeze())
plt.colorbar()
plt.show()
