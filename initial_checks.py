
import os.path
from os import path
from os import listdir
from os.path import isfile, join
import cv2



def initial_checks_func(folder_path):

    # Check folder exists
    f_exists = path.exists(folder_path)
    #print('The folder exists: ', f_exists)

    # Check folder is not empty
    isempty = len(os.listdir(folder_path) ) == 0
    #print('The folder is empty: ', isempty)

    # Format is image based
    all_images = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')) ]

    # Number of images
    #print(len(all_images))

    # Check the size of the images
    for i in all_images:
        image_path = join(folder_path, i)
        image = cv2.imread(image_path)
        #print(image.shape) # if you are curious of all the images

    return (f_exists, isempty, len(all_images), image.shape, all_images)


"""
folder_path = "/home/asoria/Documents/913440_not_localized/ID913440_images/"
print(initial_checks(folder_path))
"""    
