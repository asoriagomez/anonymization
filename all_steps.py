from detection_pipeline import *
from initial_quality_project import *
from blurred_auto import *
import pandas as pd
from xml_files import *

# DO THIS -------------------------------------------------------------------------------------------------

# Provide the project folder paths
hs = "/home/asoria/Documents/proyecto_bretagne/port_kerity/"


folder_path = "/home/asoria/Documents/proyecto_bretagne/port_kerity/sample_original_images/"
store_summary_dict = r'/home/asoria/Documents/proyecto_bretagne/port_kerity/summary_project.csv'


# UNTIL HERE ------------------------------------------------------------------------------------------------

# Create and empty dictionary for all the and results, create list of quality parameters
summary_dict = {}
params_images = ['modeHue', 'medianSat', 'medianVal', 'avgLy', 'varLy', 'skewness', 'kurtosis', 'asg', 'sobel', 'hough', 'modaLBP', 'entropy']
summary_dict['folder_name'] = folder_path


# Initial checks to see the state of the folder
print('Initial folder checks')
f_exists, isempty, n_images, shape_images, all_images = initial_checks_func(folder_path)

summary_dict['n_imgs'] = n_images
summary_dict['before'] = {}
summary_dict['before']['inputs'] = {}
summary_dict['before']['inputs']['images'] = {}

# Quality checks for each image in the project
print('Project quality checks')

(varmodeHue, varavgLys, varHough, varEntropy, img_chars) = project_description(folder_path, all_images, False, join(hs,"hist_orig.png"))
summary_dict['before']['inputs']['varmodeHue'] = varmodeHue
summary_dict['before']['inputs']['varavgLys'] = varavgLys
summary_dict['before']['inputs']['varHough'] = varHough
summary_dict['before']['inputs']['varEntropy'] = varEntropy

for image in all_images:
    summary_dict['before']['inputs']['images'][image] = {}
    summary_dict['before']['inputs']['images'][image]['img_char'] = {}

    for n in range(len(params_images)):
        p = params_images[n]
        summary_dict['before']['inputs']['images'][image]['img_char'][p] = img_chars[image][n+3]


# Obtain ground truth
print('Ground truth')
image_gt_dict = obtain_gt(folder_path, all_images)

# Run detection algorithm
print('Detection algorithm')
image_dp_dict = obtain_automatic(all_images, folder_path)

# Display performance of detection algorithm
print('Performance of detection algorithm')
detection_performance(image_dp_dict, show=False, name=join(hs,"det_perf.png"))

# Print GT and DP
print('Ground truth ', image_gt_dict)
print('Detections ', image_dp_dict)

# Obtain the parameters for the ideal object to find
print('Calculate ideal parameters')
ideal_filename = '/home/asoria/Documents/913440_not_localized/ideal_greek_image'
ideal_params = set_baseline(ideal_filename)

# Filter the detections that are very different to the ideal object
print('Filter detections')
filtered_dp_dict = check_all_detections_quality(image_dp_dict, all_images, folder_path, ideal_params)
for f in all_images:
    summary_dict['before']['inputs']['images'][f]['detections'] = {}
    ndetections = len(filtered_dp_dict[f]['keep'])
    for i in range(ndetections):
        detname = 'det'+str(i+1)
        summary_dict['before']['inputs']['images'][f]['detections'][detname] = {}
        summary_dict['before']['inputs']['images'][f]['detections'][detname]['coordis'] = filtered_dp_dict[f]['keep'][i] 


        for n in range(len(params_images)):
            p = params_images[n]
            summary_dict['before']['inputs']['images'][f]['detections'][detname][p] = filtered_dp_dict[f]['unideal'][i][n+3]

# Calculate optimum IoU for GT and DP to maximize F1 score in the project
print('IoU calculation')
th_opt = find_optim_iou(all_images, image_gt_dict, filtered_dp_dict, True)
print('Optimum threshold = ',th_opt)
summary_dict['th_opt'] = th_opt


# Calculate F1 score for each image
print('Calculate F1')
image_f1_dict = img_eval_f1score(all_images, image_gt_dict, filtered_dp_dict, th_opt, True, join(hs,"F1.png"))
for image in all_images:
    summary_dict['before']['inputs']['images'][image]['F1'] = image_f1_dict[image]


# Run blurring algorithm
print('Blurring algorithm')
image_blurred_dict, folder_path_out = blur_automatic(all_images, filtered_dp_dict, folder_path)
summary_dict['after'] = {}
summary_dict['after']['inputs'] = {}


# Display performance of blurring algorithm
print('Performance of blurring algorithm')
detection_performance(image_blurred_dict, True, join(hs,"blur_perf.png"))

# Evaluation of blurred detections
print('Evaluate blurred detections')
augmented_blurred_dict = check_all_blurred_quality(image_blurred_dict, all_images, folder_path_out)
print(augmented_blurred_dict)
summary_dict['after']['inputs']['images'] = {}

for f in all_images:
    summary_dict['after']['inputs']['images'][f] = {}
    summary_dict['after']['inputs']['images'][f]['detections'] = {}

    ndetections = len(augmented_blurred_dict[f]['keep'])
    for i in range(ndetections):
        detname = 'det'+str(i+1)
        summary_dict['after']['inputs']['images'][f]['detections'][detname] = {}
        summary_dict['after']['inputs']['images'][f]['detections'][detname]['coordis'] = augmented_blurred_dict[f]['keep'][i]

        for n in range(len(params_images)):
            p = params_images[n]
            summary_dict['after']['inputs']['images'][f]['detections'][detname][p] = augmented_blurred_dict[f]['unideal'][i][n+3]
print('Evaluate blurred images')

(varmodeHue_b, varavgLys_b, varHough_b, varEntropy_b, img_chars_b) = project_description(folder_path_out, all_images, show=False, x = join(hs,"hist_blur.png"))
summary_dict['after']['inputs']['varmodeHue'] = varmodeHue_b
summary_dict['after']['inputs']['varavgLys'] = varavgLys_b
summary_dict['after']['inputs']['varHough'] = varHough_b
summary_dict['after']['inputs']['varEntropy'] = varEntropy_b

for image in all_images:
    summary_dict['after']['inputs']['images'][image]['img_char'] = {}

    for n in range(len(params_images)):
            p = params_images[n]
            summary_dict['after']['inputs']['images'][image]['img_char'][p] = img_chars_b[image][n+3]

df6 = pd.DataFrame.from_dict(summary_dict) 
print(df6)
df6.to_csv(store_summary_dict)        
rec_print(summary_dict,0)

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

report_before_path = '/home/asoria/Documents/proyecto_bretagne/port_kerity/report_original.xml'
report_after_path = '/home/asoria/Documents/proyecto_bretagne/port_kerity/report_blurred.xml'
store_report_dict = r'/home/asoria/Documents/proyecto_bretagne/port_kerity/report_project.csv'

# UNTIL HERE

report_dict = {}

report_dict['before'] = {}
report_dict['after'] = {}

report_dict['before']['Cloud4D'] = {}
report_dict['before']['Matic4D'] = {}
report_dict['before']['CCompare'] = {}

report_dict['after']['Cloud4D'] = {}
report_dict['after']['Matic4D'] = {}
report_dict['after']['CCompare'] = {}

params_cloud = ['n_calibrated', 'GSD', 'optim', '2D_BBA', '3D_BBA', 'keypoints_img', 'matches_img', 'mre']
params_matic = ['MTP1', 'MTP2', 'MTP3', 'MTP4']
params_cc = ['RMS_register', 'avg_dist', 'sigma']


results_before_cloud = get_xml_thingies(report_before_path)
print('DONE report before path')
results_after_cloud = get_xml_thingies(report_after_path)
print('DONE results after path')

n = 0
for x in params_cloud:
    report_dict['before']['Cloud4D'][x] = results_before_cloud[n]
    report_dict['after']['Cloud4D'][x] = results_after_cloud[n]
    n+=1

for x in params_matic:
    report_dict['before']['Matic4D'][x] = input(x+' before:')
    report_dict['after']['Matic4D'][x] = input(x+' after:')

for x in params_cc:
    report_dict['before']['CCompare'] [x] = input(x+' before:')
    report_dict['after']['CCompare'] [x] = input(x+' after:')    


df2 = pd.DataFrame.from_dict(report_dict) 
df2.to_csv(store_report_dict)        
rec_print(report_dict,0)

