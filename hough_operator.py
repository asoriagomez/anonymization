
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join



def hough_operator_func(src, show=True):
       
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst = cv2.Canny(src, 100, 300, None,3)

    cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)

    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 80, None, 1, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (255,0,0), 2, cv2.LINE_AA)
    if show:
        plt.imshow(cdstP)
        plt.title('Canny edged and Hough lines')
        plt.show()
    else:
        None
    return len(linesP)

"""
filename = '/home/asoria/Documents/zita9999/car1.jpg'
hough_operator(filename)
"""
