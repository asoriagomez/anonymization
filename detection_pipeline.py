from numpy import diff
from get_ground_truth_folder import *
from initial_checks import *
from license_plate import *
from nms_algorithm import *
from confusion_matrix_calculation import *
from initial_quality_project import *

import pandas as pd


# Obtain ground truth manually
def obtain_gt(folder_path, all_images):
    
    plt.figure()
    plt.text(0.2,0.5,'Click in TL, BR for all the plates you see in an image \n and then press Enter to close the image.\n Click top right X to begin:')
    plt.title('Obtain GT instructions:')
    plt.show()
    image_gt_dict = obtain_ground_truth(folder_path, all_images)
    plt.close()
    return image_gt_dict



# Obtain detections automatically
def obtain_automatic(all_images, folder_path):
    image_dp_dict = {}
    n=-1
    for f in all_images:
        n=n+1
        filename = join(folder_path, f)
        src = cv2.imread(filename) #in BGR
        plate_img, _, levelWeights, diff_time5, plate_rects, psutil_before, psutil_after = detect_plate3(img = src, scaleF = 1.1, minNei = 3)
        plate_img_copy = plate_img.copy()
        #print('Original plate rects: ', plate_rects)
        # display(src, title='Output of openCV algorithm', keep = plate_rects)
        #print('The confidence score is given by:',levelWeights)
        
        keep = NMS(plate_rects, levelWeights)
        #print('Filtered keep: ', keep)
        """
        if f=='Image_000071.jpg':
            display(plate_img_copy, title='Output of NMS algorithm', keep=keep)
        else:
            None
        """
        image_dp_dict[f] = {'keep':keep, 'diff_time':diff_time5, 'ram_before':psutil_before, 'ram_after':psutil_after}
    return image_dp_dict



# Define IoU threshold
# Precision - recall curve for various IoU thresholds
def find_optim_iou(all_images, image_gt_dict, image_dp_dict, show, name):
    all_f1s = []
    for f in all_images:
        gt = image_gt_dict[f]
        dp = image_dp_dict[f]['keep']
        f1s, thres = calc_f1s(gt, dp)
        all_f1s.append(f1s)
        
    avg_f1s = np.mean(all_f1s, axis=0)
    pos = np.argmax(avg_f1s[::-1])
    th_opt = np.linspace(0,1,50)[::-1][pos]
    if show:
        fig, ax = plt.subplots()
        ax.plot(thres,avg_f1s)
        ax.plot(th_opt,max(avg_f1s),'ro')
        ax.set_title('Average F1s vs IoU')
        ax.set_ylabel('F1')
        ax.set_xlabel('IoU')
        plt.grid()
        plt.show()
        fig.savefig(name)
    else:
        None
    return th_opt


#Performance
def detection_performance(image_dp_dict, show, name):
    #plt.close()
    diff_times = [x['diff_time'] for x in image_dp_dict.values()]
    rams_before = [x['ram_before'] for x in image_dp_dict.values()]
    rams_after = [x['ram_after'] for x in image_dp_dict.values()]
    avg_diff_time = np.average(diff_times)
    avg_rams_before = np.average(rams_before)
    avg_rams_after = np.average(rams_after)
    if show:
        fig, ax1 = plt.subplots()
        fig.set_figheight(10)
        fig.set_figwidth(16)
        color = 'tab:red'
        ax1.set_ylabel('Wall clock time (s)', color = color)
        ax1.plot(list(image_dp_dict.keys()), diff_times, color = color, label = 'Avg diff time = '+str(avg_diff_time))
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticklabels(labels = list(image_dp_dict.keys()), rotation = 45)
        ax1.legend(loc = 'upper left')

        ax2 = ax1.twinx()
        color = 'b'
        ax2.set_ylabel('RAM consumption (%)', color = color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.plot(list(image_dp_dict.keys()), rams_after, color='mediumblue',label =  'Avg RAM after = '+str(avg_rams_after))
        ax2.plot(list(image_dp_dict.keys()), rams_before, 'lightsteelblue', label = 'Avg RAM before = '+str(avg_rams_before))
        ax2.legend(loc = 'upper right')
        fig.tight_layout() 
        plt.grid()
        plt.suptitle('Performance  algorithm')
        plt.tight_layout()
        fig.savefig(name)
        plt.show()
        #plt.close()
    else:
        None
    return (diff_times, rams_before, rams_after)



# Evaluation of detections with quality parameters
# 0) What is a good detection
def is_good(ideal_params, unideal_params):
    thresholds = [30, 30, 30, 20, 20, 100, 100, 400, 100, 150, 50, 200]
    wrong_params = 0
    for i in range(len(thresholds)):
        i=i+3
        ideal = ideal_params[i]
        unideal = unideal_params[i]
        thres = thresholds[i-3]
        discrepance = abs(ideal-unideal)/abs(ideal)
        if discrepance>thres:
            wrong_params+=1
        else:
            None
    #You need 5 or more parameters wrong to discard the detection for example
    return (wrong_params<5)


# 1) Set baseline of what you are looking for
def set_baseline(ideal_filename):
    ideal_image = cv2.imread(ideal_filename)
    
    ideal_params =  params_one_array(ideal_image, 'Ideal parameters', show=False, print_all=False)
    ii = cv2.cvtColor(ideal_image, cv2.COLOR_BGR2RGB)
    """
    plt.imshow(ii)
    plt.title('Ideal detection')
    """
    return ideal_params



# 2) Calculate parameters for all the detections and see if they are 'good'
def check_all_detections_quality(image_dp_dict, all_images, folder_path, ideal_params):
    filtered_dp_dict = image_dp_dict.copy()
    g = 0
    b = 0
    for f in all_images:
        filename = join(folder_path, f)
        src = cv2.imread(filename)
        copy_src = src.copy()
        detections = image_dp_dict[f]['keep']
        #print(f+" : "+detections)

        i=0
        filtered = []
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
            unideal_params = params_one_array(zoom_img, f+' Detection = '+str(i), show=False, print_all=False)

            # 3) Check if the detections comply with what you are looking for
            if is_good(ideal_params, unideal_params):
                g = g+1
                filtered.append([x,y,w,h])
                unid_par.append(unideal_params)
            else:
                b = b+1
                None

            filtered_dp_dict[f]['keep'] = filtered
            filtered_dp_dict[f]['unideal'] = unid_par
    print('Number of detections considered good = ',str(g))
    print('Number of detections considered bad = ', str(b))
    return filtered_dp_dict


def xx():
    print('a')


# Evaluation of each image with F1 score
def img_eval_f1score(all_images, image_gt_dict, other_dict, th_opt, show, name):
    plt.close()
    image_f1_dict = {}

   

    for f in all_images:
        
        image_f1_dict[f] = {}
        

        gt = image_gt_dict[f]
        dp = other_dict[f]['keep'] # filtered_dp_dict or image_dp_dict
        TN, TP, FN, FP, accuracy, precision, recall, f1 = confusion_matrix_calc(gt, dp, th_opt)
        image_f1_dict[f]['TP'] = TP
        image_f1_dict[f]['FN'] = FN
        image_f1_dict[f]['FP'] = FP
        image_f1_dict[f]['recall'] = recall
        image_f1_dict[f]['precision'] = precision
        image_f1_dict[f]['f1'] = f1


    return image_f1_dict


def plot_confusion_values(image_f1_dict, name):
    df = pd.DataFrame(image_f1_dict)

    tpavg = np.average(df.loc['TP'])
    fnavg = np.average(df.loc['FN'])
    fpavg = np.average(df.loc['FP'])

    recallavg = np.average(df.loc['recall'])
    precisionavg = np.average(df.loc['precision'])
    f1avg = np.average(df.loc['f1'])

    fig, ax1 = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(16)
    color = 'tab:red'
    ax1.set_ylabel('Confusion matrix', color = color)
    ax1.plot(df.keys(), df.loc['TP'], color='red', label = 'TP avg = '+str(tpavg))
    ax1.plot(df.keys(), df.loc['FN'], color='black', label = 'FN avg = '+str(fnavg))
    ax1.plot(df.keys(), df.loc['FP'], color='magenta', label = 'FP avg = '+str(fpavg))

    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticklabels(labels = list(image_f1_dict.keys()), rotation = 45)
    ax1.legend(loc = 'upper left')

    ax2 = ax1.twinx()
    color = 'b'
    ax2.set_ylabel('Parameters', color = color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.plot(df.keys(), df.loc['recall'], color='cyan', label = 'Recall avg = '+str(recallavg))
    ax2.plot(df.keys(), df.loc['precision'], color='green', label = 'Precision avg = '+str(precisionavg))
    ax2.plot(df.keys(), df.loc['f1'], color='blue', label = 'f1 avg = '+str(f1avg))
    ax2.legend(loc = 'upper right')
    fig.tight_layout() 
    plt.grid()
    plt.suptitle('Performance  algorithm')
    plt.tight_layout()
    fig.savefig(name)
    plt.show()
    plt.close()
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
Now try all the whole detection pipeline


folder_path = "/home/asoria/Documents/913440_not_localized/ID913440_images/"
_, _, _, _, all_images = initial_checks_func(folder_path)

image_gt_dict = obtain_gt(folder_path, all_images)
image_dp_dict = obtain_automatic(all_images, folder_path)
print('Ground truth ', image_gt_dict)
print('Detections ', image_dp_dict)

th_opt = find_optim_iou(all_images, image_gt_dict, image_dp_dict, True)


ideal_filename = '/home/asoria/Documents/913440_not_localized/ideal_greek_image'
ideal_params = set_baseline(ideal_filename)
filtered_dp_dict = check_all_detections_quality(image_dp_dict, all_images, folder_path, ideal_params)
image_f1_dict = img_eval_f1score(all_images, image_gt_dict, filtered_dp_dict, th_opt, True)

detection_performance(filtered_dp_dict, show=True)
"""