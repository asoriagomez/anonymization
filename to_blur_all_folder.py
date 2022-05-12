from cv2 import blur
from license_plate import *
from nms_algorithm import *

from os import listdir
from os.path import isfile, join

def blur_black(filename_in, filename_out):
    image_in = cv2.imread(filename_in)
    scaleF = 1.1
    minNei = 1
    image_detected, reject_levels, level_weights, diff_time,  detected_rectangles = detect_plate3(image_in, scaleF, minNei)
    print(detected_rectangles)
    keep = NMS(detected_rectangles, level_weights)
    print(keep)
    image_blurred = detect_blur(image_in, keep)
    cv2.imwrite(filename_out, image_blurred)



mypath_in = '/home/asoria/Documents/zita9999/ID1055745/'
mypath_out = '/home/asoria/Documents/zita9999/ID1055745_blurred/'

onlyfiles = [f for f in listdir(mypath_in) if isfile(join(mypath_in, f))]

for f in onlyfiles:
    filename_in = join(mypath_in, f)
    filename_out = join(mypath_out, f)
    blur_black(filename_in, filename_out)


