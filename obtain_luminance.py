
import cv2
import numpy as np


name = "car1"
name_open = name+".jpg"
def log_avg_var_luminance(name_open):
        
    # Read in BGR
    plate = cv2.imread("/home/asoria/Documents/zita9999/"+name_open)

    if not plate.shape[2]==3:
        raise ValueError('The image is not RGB.')

    B=plate[:,:,0]
    G=plate[:,:,1]
    R=plate[:,:,2]

    if not R.shape==G.shape==B.shape!=0:
        raise ValueError('R,G,B are not same size or are zero')

    # According to ITU BT.709
    Y = 0.2126*R + 0.7152*G + 0.0722*B

    # To avoid ln(0)
    delta = 10*np.exp(-5)

    # Work with the ln of the luminance
    Ly = np.log(delta+Y)

    # a) Obtain the exp(average(ln(Y)))
    av = np.average(Ly)
    avgLy = np.exp(av)
    print("The log-average luminance of the image is = ","{:.2f}".format(avgLy))

    # a) Obtain the exp(variance(ln(Y)))
    v = np.var(Ly)
    varLy = np.exp(v)
    print("The log-variance luminance of the image is = ","{:.2f}".format(varLy))
    return avgLy, varLy

avgLy, varLy = log_avg_var_luminance(name_open)
print(avgLy, varLy)