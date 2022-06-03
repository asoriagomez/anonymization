from get_ground_truth_folder import *
from initial_checks import *
from license_plate import *
from nms_algorithm import *
from confusion_matrix_calculation import *
from initial_quality_project import *
from detection_pipeline import *


def blur_automatic(all_images, filtered_dp_dict, folder_path_in):
    folder_path_out = join(folder_path_in,'blurred/')
    image_blurred_dict = {}
    for f in all_images:
        destination = join(folder_path_out,f)
        filename = join(folder_path_in, f)
        src = cv2.imread(filename) #in BGR
        plate_img, delay_blur, psutil_before, psutil_after = detect_blur(src, filtered_dp_dict[f]['keep'])

        display(plate_img, title='Output of blurring algorithm', keep=filtered_dp_dict[f]['keep'])
        image_blurred_dict[f] = {'diff_time': delay_blur, 'ram_before':psutil_before, 'ram_after':psutil_after, 'keep':filtered_dp_dict[f]['keep']}
        cv2.imwrite(destination, plate_img)
    return image_blurred_dict, folder_path_out

# 2) Calculate parameters for all the detections and see if they are 'good'
def check_all_blurred_quality(image_blurred_dict, all_images, folder_path_blurred):
    augmented_blurred_dict = image_blurred_dict.copy()

    for f in all_images:
        filename = join(folder_path_blurred, f)
        src = cv2.imread(filename)
        copy_src = src.copy()
        detections = image_blurred_dict[f]['keep']
        print(detections)

        i=0
        unid_par = []
        for (x,y,w,h) in detections:
            i+=1
            # Find detections
            x_offset = x
            y_offset = y
            x_end = x+w
            y_end = y+h
            zoom_img = copy_src[y_offset:y_end, x_offset:x_end]

            # Start evaluating quality parameters
            unideal_params = params_one_array(zoom_img, f+' Detection blurred = '+str(i), show=False, print_all=False)
            unid_par.append(unideal_params)

            augmented_blurred_dict[f]['unideal'] = unid_par
    return augmented_blurred_dict
# -------------------------------------------------------------

"""
Now try all the whole detection pipeline

image_blurred_dict = blur_automatic(all_images, filtered_dp_dict, folder_path)
detection_performance(image_blurred_dict, True)
print('Blur automatic')
"""