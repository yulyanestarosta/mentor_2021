import cv2
import numpy as np
import matplotlib.pyplot as plt

img_1 = cv2.imread('photo_2021-03-03 17.22.37.jpeg', cv2.IMREAD_GRAYSCALE)
img_2 = cv2.imread('photo_2021-03-03 17.22.32.jpeg', cv2.IMREAD_GRAYSCALE)

f = np.fft.fft2(img_1)
b = np.fft.fft2(img_2)
magnitude_spectrum_f = 20*np.log(np.abs(f))
f_shift = np.fft.fftshift(f)


phase_spectrum_f = np.angle(f_shift)

magnitude_spectrum_f = np.asarray(magnitude_spectrum_f, dtype=np.uint8)
magnitude_spectrum_b = 20*np.log(np.abs(b))
magnitude_spectrum_b = np.asarray(magnitude_spectrum_b, dtype=np.uint8)
combined_1 = np.multiply(np.abs(f), np.exp(1j*np.angle(b)))
imgCombined_1 = np.real(np.fft.ifft2(combined_1))
plt.imshow(imgCombined_1, cmap='Blues', label='imgCombined_1')

plt.show()
