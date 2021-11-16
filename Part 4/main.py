import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

xfile = open('xdata.txt')
x_arr = np.empty(1000000)
i = 0
for x in xfile:
    x_arr[i] = x
    i += 1

yfile = open('ydata.txt')
y_arr = np.empty(1000000)
i = 0
for y in yfile:
    y_arr[i] = y
    i += 1


def func(x, a1, b1, c1):
    return a1 * np.exp(-((x - b1) / c1) ** 2)


popt, pcov = opt.curve_fit(func, x_arr, y_arr)
plt.plot(x_arr, y_arr, 'o', color='k', markersize=0.5)
plt.plot(x_arr, func(x_arr, popt[0], popt[1], popt[2]))
print(popt)
plt.show()

#
# def GaussSum(x, *p):
#     n = len(p) / 3
#     A = p[:n]
#     w = p[n:2 * n]
#     c = p[2 * n:3 * n]
#     y1 = sum(
#         [A[i] * np.exp(-(x - c[i]) ** 2. / (2. * (w[i]) ** 2.)) / (2 * np.pi * w[i] ** 2) ** 0.5 for i in range(n)])
#     return y1
#
#
# params = [1., 1., -3.]  # parameters for a single gaussian
#
# popt1, pcov1 = curve_fit(GaussSum, xfile, yfile, p0=params)

# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.stats
# import scipy.optimize
#
# data = np.array([-69,3, -68, 2, -67, 1, -66, 1, -60, 1, -59, 1,
#                  -58, 1, -57, 2, -56, 1, -55, 1, -54, 1, -52, 1,
#                  -50, 2, -48, 3, -47, 1, -46, 3, -45, 1, -43, 1,
#                  0, 1, 1, 2, 2, 12, 3, 18, 4, 18, 5, 13, 6, 9,
#                  7, 7, 8, 5, 9, 3, 10, 1, 13, 2, 14, 3, 15, 2,
#                  16, 2, 17, 2, 18, 2, 19, 2, 20, 2, 21, 3, 22, 1,
#                  24, 1, 25, 1, 26, 1, 28, 2, 31, 1, 38, 1, 40, 2])
# x, y = data.reshape(-1, 2).T
#
# def tri_norm(x, *args):
#     m1, m2, m3, s1, s2, s3, k1, k2, k3 = args
#     ret = k1*scipy.stats.norm.pdf(x, loc=m1 ,scale=s1)
#     ret += k2*scipy.stats.norm.pdf(x, loc=m2 ,scale=s2)
#     ret += k3*scipy.stats.norm.pdf(x, loc=m3 ,scale=s3)
#     return ret
#
#
# params = [-50, 3, 20, 1, 1, 1, 1, 1, 1]
#
# fitted_params,_ = scipy.optimize.curve_fit(tri_norm,x, y, p0=params)
#
# plt.plot(x, y, 'o')
# xx = np.linspace(np.min(x), np.max(x), 1000)
# plt.plot(xx, tri_norm(xx, *fitted_params))
# plt.show()

# from matplotlib import pyplot as plt
# from astroML.sum_of_norms import sum_of_norms, norm
# import numpy as np
#
# xfile = open('xdata.txt')
# x = np.empty(1000000)
# i = 0
# for x1 in xfile:
#     x[i] = x1
#     i += 1
#
# yfile = open('ydata.txt')
# y = np.empty(1000000)
# i = 0
# for y1 in yfile:
#     y[i] = y1
#     i += 1
# # truncate the spectrum
# mask = (x >= 1) & (x < 10000000)
# x = x[mask]
# y = y[mask]
#
# for n_gaussians in (8, 8, 8):
#     # compute the best-fit linear combination
#     w_best, rms, locs, widths = sum_of_norms(x, y, n_gaussians,
#                                              spacing='linear',
#                                              full_output=True)
#
#     norms = w_best * norm(x[:, None], locs, widths)
#
#     # plot the results
#     plt.figure()
#     plt.plot(x, y, '-k', label='input spectrum')
#     ylim = plt.ylim()
#
#     plt.plot(x, norms, ls='-', c='#FFAAAA')
#     plt.plot(x, norms.sum(1), '-r', label='sum of gaussians')
#     plt.ylim(-0.1 * ylim[1], ylim[1])
#
#     plt.legend(loc=0)
#
#     plt.text(0.97, 0.8,
#              "rms error = %.2g" % rms,
#              ha='right', va='top', transform=plt.gca().transAxes)
#     plt.title("Fit to a Spectrum with a Sum of %i Gaussians" % n_gaussians)
#
# plt.show()
