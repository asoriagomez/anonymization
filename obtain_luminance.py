from PIL import Image
import cv2
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------
# Luminance

def aux_log_avg_var_luminance(array):
    """
    Inputs
    - np.int32 array which can be (a piece of) an image in grayscale

    Outputs:
    - Luminance of the grayscale array
    """

    delta = 10*np.exp(-5) # To avoid ln(0)
    Ly = np.log(delta+array) # Work with the ln of the luminance

    # Obtain the exp(average(ln(Y)))
    av = np.average(Ly)
    avgLy = np.exp(av)

    # Obtain the exp(variance(ln(Y)))
    v = np.var(Ly)
    varLy = np.exp(v)
    return (avgLy, varLy)


def log_avg_var_luminance(filename,plate_rects = []):
    """
    Inputs:
    - Path to an image (compulsory)
    - Array with detected plates (optional)
    
    Outputs:
    - Array with (avgLuminance, varLuminance) for each license plate, or whole image if none was provided
    """

    im = Image.open(filename).convert('L') # to grayscale
    array_im = np.asarray(im, dtype=np.int32)

    if len(plate_rects) == 0:
        print('Empty plate_rects, calculate luminance for whole image')
        plate_rects = [(0,0,array_im.shape[-1],array_im.shape[0])]

    else:
        print('Calculate luminance for each blurred rectangle')

    all_luminances = []
    nsec = 0
    for (x,y,w,h) in plate_rects:
        nsec+=1
        print("x=",x,"y=",y,"w=", w, "h=",h)
        x_offset = x
        y_offset = y
        
        x_end = x+w
        y_end = y+h
        
        #getting the points that show the license plate
        array_section = array_im[y_offset:y_end, x_offset:x_end]
        (avgLum, varLum) = aux_log_avg_var_luminance(array_section)
        all_luminances.append((avgLum, varLum))
    return all_luminances


    return avgLy, varLy


# ------------------------------------------------------------------------------------------------------------- #
# Trying out the code with different parameters 

name = "car1_blurredX_1.1_nei_3.png"
filename = "/home/asoria/Documents/zita9999/"+name
plate_rects = [[1688 , 235 , 129  , 43],
 [1711 , 233 , 158  , 53],
 [ 450 , 421 , 1050 , 350],
 [ 503 , 587  , 73 ,  24],
 [ 663 , 526 , 324 , 108],
 [  26 , 552 , 178 ,  59]]
all_luminances = log_avg_var_luminance(filename)#, plate_rects)
print('blurred=',all_luminances)