from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters.rank import gradient
from skimage.util import img_as_ubyte
import cv2
import warnings
warnings.filterwarnings('ignore')
from os.path import isfile, join



# ----------------------------------------------------------------------------------------------------------------------------
# AVERAGE OF SQUARED GRADIENTS "
# compute sharpness as the average of the gradients in 2D
# it is a naive approach but at least works


def aux_calculate_sharpness(array, show=True):
    """
    Inputs
    - np.int32 array which can be (a piece of) an image in grayscale
    - String to put as a title for the visualization

    Outputs:
    - Average of squared gradients of the grayscale array
    - Visualization of the gradients of the array
    """
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

    gy, gx = np.gradient(array)
    gnorm = np.sqrt(gx**2 + gy**2)
    sharpness = np.average(gnorm)

    #to display a nice visual result, but which is then not used for computation
    selection_element = disk(2) # matrix of n pixels with a disk shape
    cat_sharpness = gradient(array, selection_element) 
    if show:
        plt.imshow(cat_sharpness)
        plt.axis('off')
        plt.colorbar()
        plt.title('Average of square gradients')
        plt.show()
    else:
        None
    return sharpness


def calculate_avg_sq_gradients(filename,plate_rects = []):
    """
    Inputs:
    - Path to an image (compulsory)
    - Array with detected plates (optional)
    
    Outputs:
    - Blue-greenish images for visualization which you should go closing to continue running the algorithm
    - Array with sharpness for each license plate, or whole image if none was provided
    """

    im = Image.open(filename).convert('L') # to grayscale
    array_im = np.asarray(im, dtype=np.int32)

    if len(plate_rects) == 0:
        #print('Empty plate_rects, calculate ASG for whole image')
        plate_rects = [(0,0,array_im.shape[-1],array_im.shape[0])]

    else:
        None
        #print('Calculate sharpness for each blurred rectangle')

    all_sharpness = []
    nsec = 0
    for (x,y,w,h) in plate_rects:
        nsec+=1
        #print("x=",x,"y=",y,"w=", w, "h=",h)
        x_offset = x
        y_offset = y
        
        x_end = x+w
        y_end = y+h
        
        #getting the points that show the license plate
        array_section = array_im[y_offset:y_end, x_offset:x_end]
        plt.imshow(array_section, cmap='gray')
        plt.title('Grayscale image')
        plt.show()

        sharpness = aux_calculate_sharpness(array_section,title =' section: '+str(nsec))
        all_sharpness.append(sharpness)
        #print(all_sharpness)
    return all_sharpness



# ------------------------------------------------------------------------------------------------------------- #
# Trying out the code with different parameters 
"""

name = "car1.jpg"#_blurredX_1.1_nei_3.png"
filename = "/home/asoria/Documents/zita9999/"+name
plate_rects = [[1688 , 235 , 129  , 43],
 [1711 , 233 , 158  , 53],
 [ 450 , 421 , 1050 , 350],
 [ 503 , 587  , 73 ,  24],
 [ 663 , 526 , 324 , 108],
 [  26 , 552 , 178 ,  59]]
 
#plate_rects = []

calculate_avg_sq_gradients(filename, plate_rects)
"""