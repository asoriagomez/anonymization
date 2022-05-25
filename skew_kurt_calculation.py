
import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from scipy.stats import skew, kurtosis

def skewness_kurtosis(src, show=True):

    grayed = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    fl = grayed.flatten()

    skewness = skew(fl)
    kurt = kurtosis(fl)
    if show:
        plt.imshow(grayed, cmap='gray')
        plt.title('Grayscale image')
        plt.show()
        cm = plt.cm.get_cmap('gist_gray')
        n, bins, patches = plt.hist(grayed.flatten(), 30)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = bin_centers - min(bin_centers)
        col /= max(col)

        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))
        plt.title("Grayscale histogram")
        plt.show()
    else:
        None
    return (skewness, kurt)


"""

filename = '/home/asoria/Documents/zita9999/car1.jpg'
print(skewness_kurtosis(filename))

"""
