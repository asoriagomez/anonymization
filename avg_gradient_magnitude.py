from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters.rank import gradient
from skimage.util import img_as_ubyte

import warnings
warnings.filterwarnings('ignore')

#compute sharpness as the average of the gradients in 2D
#it is a naive approach but at least works

name = "car1_blurred.png"
filename = "/home/asoria/Documents/zita9999/"+name

def calculate_sharpness(array):
    "input a np.int32 array which can be a piece of an image in grayscale"
    gy, gx = np.gradient(array)
    gnorm = np.sqrt(gx**2 + gy**2)
    sharpness = np.average(gnorm)

    #to display a nice visual result, but which is then not used for computation
    selection_element = disk(2) # matrix of n pixels with a disk shape
    cat_sharpness = gradient(array, selection_element)

    plt.imshow(cat_sharpness, cmap="viridis")
    plt.axis('off')
    plt.colorbar()
    plt.title(name)
    plt.show()
    return sharpness

#calculate_sharpness(filename)


plate_rects = [[691,530,299,100],[36,551,156,52],[498,443,980,327]]
"""
plate_rects = [[1711 , 233 , 158 ,  53],
 [1688 , 235 , 129  , 43],
 [ 450 , 421 ,1050 , 350],
 [ 503 , 587  , 73 ,  24],
 [ 663 , 526  ,324 , 108],
 [  26 , 552  ,178 ,  59]]
"""

def calc_sharpness_image(filename, plate_rects):

    im = Image.open(filename).convert('L') # to grayscale
    array_im = np.asarray(im, dtype=np.int32)


    for (x,y,w,h) in plate_rects:
        print("x=",x,"y=",y,"w=", w, "h=",h)
        x_offset = x
        y_offset = y
        
        x_end = x+w
        y_end = y+h
        
        #getting the points that show the license plate
        array_section = array_im[y_offset:y_end, x_offset:x_end]
        sharpness = calculate_sharpness(array_section)
        print(sharpness)
        
"You should go closing the images that pop up in order to continue running the algorithm"
# It is lame but I will improve it for more images, this is just a simple measure of 
# sharpness of the detected plates
calc_sharpness_image(filename, plate_rects)