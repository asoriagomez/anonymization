


import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


def calculate_sobel(src, show=True):

    image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    resized = image#cv2.resize(image,(1440, 1920))
    denoised = cv2.GaussianBlur(resized, (5,5), 0)
    sobel = cv2.Sobel(denoised, cv2.CV_16UC1, 1, 1, ksize=3, scale = 5, delta = 5, borderType=cv2.BORDER_DEFAULT)
    
    param_1 = 10 #actual threshold
    param_2 = 255 #value to put over threshold
    param_3 = cv2.THRESH_BINARY#_INV #+ cv2.THRESH_OTSU extra flags
    thresholded = cv2.threshold(sobel, param_1, param_2, param_3)[1]
    area = thresholded.shape[0]*thresholded.shape[1]
    edges = np.sum(thresholded)

    sharpness = edges/area
    #print('sharpnes = ', str(sharpness))
    if show:
        plt.imshow(thresholded, cmap='gray')
        plt.title('Sharpness with Sobel operator')
        plt.show()
    else:
        None

    return sharpness
"""

mypath_in = '/home/asoria/Documents/zita9999/detected_plates_graycar/after_blurring/'
onlyfiles = [f for f in listdir(mypath_in) if isfile(join(mypath_in, f))]

sharpnesses = {}
for f in onlyfiles:
    filename = join(mypath_in, f)
    sh = calculate_sobel(filename)
    sharpnesses[f] = sh
print(sharpnesses)

"""


