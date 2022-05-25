from re import S
import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt

from skimage import data
from skimage.color import rgb2hsv



def hsv_color(src, show_all=True, show_hist=False):
    #src = cv2.imread(filename) #in BGR
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

    rgb_img = src
    hsv_img = rgb2hsv(rgb_img)
    hue_img = hsv_img[:, :, 0]
    sat_img = hsv_img[:, :, 1]
    value_img = hsv_img[:, :, 2]

    plot_hsv_histograms(hue_img, sat_img, value_img) if show_all else None
    plot_only_histograms(hue_img, sat_img, value_img) if show_hist else None

    return (hue_img, sat_img, value_img, np.bincount((hue_img*255).flatten().astype('int64')).argmax(), np.median(sat_img), np.median(value_img*255))

def plot_hsv_histograms(hue_img, sat_img, value_img):

    cm = plt.cm.get_cmap('hsv')
    sat_cm = plt.cm.get_cmap('gray')

    fig, ax = plt.subplots(ncols=3, nrows = 2, figsize=(15, 8))

    ax[0][0].imshow(hue_img, cmap='hsv')
    ax[0][0].set_title("Hue channel")
    ax[0][0].axis('off')

    ax[0][2].imshow(value_img*255, cmap = 'gray')
    ax[0][2].set_title("Value channel")
    ax[0][2].axis('off')

    n, bins, patches = ax[1][0].hist(hue_img.flatten()*255, 30)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    ax[1][0].set_title("Hue channel histogram")


    bin_max = np.where(n == n.max())[0][0]
    norm_max = bin_max/29
    color_max = cm(norm_max)
    lin = np.linspace(0,1,30)
    org_cm = []

    for l in lin:
        tup = (color_max[0], color_max[1], color_max[2], l)
        org_cm.append(tup)


    n, bins, patches = ax[1][1].hist(sat_img.flatten(), 30)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)

    for c, p in zip(col, patches):
        index = int(np.floor(c*29))
        plt.setp(p, 'facecolor', org_cm[index])
    ax[1][1].set_title("Saturation image histogram")

    n, bins, patches = ax[1][2].hist(255*value_img.flatten(), 30)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', sat_cm(c))
    ax[1][2].set_title("Value channel histogram")


    ax[0][1].imshow(sat_img, cmap = 'Blues')
    ax[0][1].set_title("Saturation image")
    ax[0][1].axis('off')

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.show()

def plot_only_histograms(hue_img, sat_img, value_img):

    cm = plt.cm.get_cmap('hsv')
    sat_cm = plt.cm.get_cmap('gray')
    #org_cm = plt.cm.get_cmap('Oranges')

    fig, ax = plt.subplots(ncols=3, nrows = 1, figsize=(15, 4))

    n, bins, patches = ax[0].hist(hue_img.flatten()*255, 30)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    ax[0].set_title("Hue channel histogram")


    bin_max = np.where(n == n.max())[0][0]
    norm_max = bin_max/29
    color_max = cm(norm_max)
    lin = np.linspace(0,1,30)
    org_cm = []

    for l in lin:
        tup = (color_max[0], color_max[1], color_max[2], l)
        org_cm.append(tup)


    n, bins, patches = ax[1].hist(sat_img.flatten(), 30)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)

    for c, p in zip(col, patches):
        index = int(np.floor(c*29))
        plt.setp(p, 'facecolor', org_cm[index])
    ax[1].set_title("Saturation image histogram")

    n, bins, patches = ax[2].hist(value_img.flatten(), 30)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', sat_cm(c))
    ax[2].set_title("Value channel histogram")

    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    plt.show()

"""
filename = '/home/asoria/Documents/913440_not_localized/ID913440_images/Image_000039.jpg'
hue_img, sat_img, value_img = hsv_color(filename)
plot_hsv_histograms(hue_img, sat_img, value_img, filename)
"""

