import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_coordinates(filename):
    """
    Inputs:
    - Path to an image
    Outputs:
    - list of coordinates of the top-left, bottom-right points
    """

    left_clicks = list() #store coordinates

    #this function will be called whenever the mouse is right-clicked
    def mouse_callback(event, x, y, flags, params):

        if event == 1: #left click
            left_clicks.append([x, y])
            print(left_clicks)


    image = cv2.imread(filename)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', image.shape[0]*2,image.shape[1]*2)

    cv2.imshow('image', image)
    cv2.setMouseCallback('image', mouse_callback)
    
    cv2.waitKey(0) #press any key to stop the algorithm
    return left_clicks

"""

# -------------------------------------------------------------------------------
# Try algorithm with some parameters
name = "car1"
name_open = name+".jpg"
filename = "/home/asoria/Documents/zita9999/"+name_open

coordis = get_coordinates(filename)
print(coordis)
"""