
from calendar import c
import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from skimage.io import imread, imshow
from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv

def entropy_operator(src, show=True):


    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    entropy_image = entropy(src, disk(3))
    # 1 would be too sharp
    # 9 would be too blurry
    if show:
            
        plt.imshow(entropy_image, cmap='magma')
        plt.colorbar()
        plt.title('Shanon entropy')
        plt.show()
    else:
        None
    #more robust to outliers
    m = np.median(entropy_image)
    return(m)
    

"""
filename = '/home/asoria/Documents/zita9999/car1.jpg'

print(entropy_operator(filename))
"""
