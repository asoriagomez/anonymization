from license_plate import *
from nms_algorithm import *

from os import listdir
from os.path import isfile, join

def blur_all(filename_in, filename_out):
    image_in = cv2.imread(filename_in)
    scaleF = 1.05
    minNei = 1
    image_detected, reject_levels, level_weights, diff_time,  detected_rectangles = detect_plate3(image_in, scaleF, minNei)
    #print(detected_rectangles)
    keep = NMS(detected_rectangles, level_weights)
    #print(keep)
    image_blurred = detect_blur(image_in, keep)
    #print(image_blurred[0])
    cv2.imwrite(filename_out, image_blurred[0])



mypath_in = '/home/asoria/Documents/alicia_blurring_openCV/ID1056886_red_pickup/clean_images/'
mypath_out = '/home/asoria/Documents/alicia_blurring_openCV/ID1056886_red_pickup/blurred_images/'

onlyfiles = [f for f in listdir(mypath_in) if isfile(join(mypath_in, f))]
print(len(onlyfiles))
i = 0
for f in onlyfiles:
    i = i+1
    print(i)
    filename_in = join(mypath_in, f)
    filename_out = join(mypath_out, f)
    blur_all(filename_in, filename_out)


