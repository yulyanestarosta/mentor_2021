import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('photo_2021-03-03 17.22.32.jpeg', cv2.IMREAD_GRAYSCALE)
f1 = np.fft.fft2(img1)
fshift1 = np.fft.fftshift(f1)
magnitude_spectrum1 = 20*np.log(np.abs(f1))
magnitude_spectrum1 = np.asarray(magnitude_spectrum1, dtype=np.uint8)
phase_spectrum1 = np.angle(fshift1)

img2 = cv2.imread('photo_2021-03-03 17.22.37.jpeg', cv2.IMREAD_GRAYSCALE)
f2 = np.fft.fft2(img2)
fshift2 = np.fft.fftshift(f2)
magnitude_spectrum2 = 20*np.log(np.abs(f2))
magnitude_spectrum2 = np.asarray(magnitude_spectrum2, dtype=np.uint8)
phase_spectrum2 = np.angle(fshift2)

display1 = np.real(np.fft.ifft2(np.multiply(np.abs(magnitude_spectrum2), np.exp(1j * np.angle(fshift1)))))

plt.imshow(display1, cmap='gray')
plt.show()

# cv2.imshow('kek', magnitude_spectrum1)
# cv2.imshow('lol', phase_spectrum1)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
