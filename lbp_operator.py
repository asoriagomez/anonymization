
import cv2
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from skimage import feature
import numpy as np


def calculate_lbp(filename, f='Figure'):
    
    image = cv2.imread(filename)
    num_points = 24
    radius = 3
    
    grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(grayed, num_points, radius, method = 'uniform') # or nri_uniform to see TL, TR, BL, BR corners

    avg_lbp = np.mean(lbp)
    median_lbp = np.median(lbp)
    min_lbp = np.min(lbp)
    max_lbp = np.max(lbp)
    moda = np.bincount(lbp.ravel().astype('int64')).argmax()
    print('moda = ', str(moda))
    print('median_lbp = ', str(median_lbp))

    plt.imshow(lbp, cmap='gray')
    plt.title('Texture with LBP operator '+f)
    plt.show()

    plt.hist(lbp.ravel())
    plt.title('LBP histogram '+f)
    plt.show()

    return avg_lbp, median_lbp, min_lbp, max_lbp, moda

mypath_in = '/home/asoria/Documents/zita9999/detected_plates_graycar/after_blurring/'
onlyfiles = [f for f in listdir(mypath_in) if isfile(join(mypath_in, f))]

textures = {}
for f in onlyfiles:
    filename = join(mypath_in, f)
    avg_lbp, median_lbp, min_lbp, max_lbp, moda = calculate_lbp(filename,f)
    textures[f] = (avg_lbp, median_lbp, min_lbp, max_lbp, moda)


print(textures)

a = np.array([[0,0,0],[1,1,0],[1,1,0]])
lbp_a=feature.local_binary_pattern(a, 8, 1, method = 'uniform')
print(lbp_a)
plt.imshow(lbp_a, cmap='gray')
plt.title('Texture with LBP operator ')
plt.show()
plt.hist(lbp_a.ravel())
plt.title('LBP histogram ')
plt.show()