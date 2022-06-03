from get_pixel_coordinates import *
from initial_checks import *


def obtain_ground_truth(folder_path, all_images):

    image_gt_dict = {}

    for f in all_images:
        # Press enter if you don't see any plate or if you are finished
        # Left-click Top Left (an average between both Tops and both Lefts) and Bottom Right (an average between both Bottoms and both Rights)

        filename = join(folder_path, f)
        coordis = get_coordinates(filename) # [[TL, BR], [TL2, BR2], ...]
        image_gt_dict[f] = coordis
    

    return image_gt_dict


"""
folder_path = "/home/asoria/Documents/913440_not_localized/ID913440_images/"

# Check folder and check images ------------------------------------------------------------------------------------------------------------------------
f_exists, is_empty, len_allimages, i_shape, all_images  = initial_checks_func(folder_path)
print('Folder exists:', f_exists, ', and there are', len_allimages,'images of resolution:', i_shape[0],'px,',i_shape[1], 'px and',i_shape[2],'BGR color spaces.')

image_gt_dict = obtain_ground_truth(all_images)
print(image_gt_dict)
"""
