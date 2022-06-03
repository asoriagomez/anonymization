
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
    len_allimages = len(all_images)
    i_shape = image.shape
    print('Folder exists:', f_exists, ', and there are', len_allimages,'images of resolution:', i_shape[0],'px,',i_shape[1], 'px and',i_shape[2],'BGR color spaces.')

    return (f_exists, isempty, len(all_images), image.shape, all_images)


"""
folder_path = "/home/asoria/Documents/913440_not_localized/ID913440_images/"
f_exists, is_empty, len_allimages, i_shape, all_images  = initial_checks_func(folder_path)
print('Folder exists:', f_exists, ', and there are', len_allimages,'images of resolution:', i_shape[0],'px,',i_shape[1], 'px and',i_shape[2],'BGR color spaces.')

"""    
