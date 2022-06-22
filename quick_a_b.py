from detection_pipeline import *
from initial_quality_project import *
from blurred_auto import *
import pandas as pd
from xml_files import *
from visualization import *

from matplotlib.pyplot import show, plot



def q_analyze_blur(hs, folder_path, store_summary_dict, report_before_path, report_after_path, store_report_dict):
        
    # Create and empty dictionary for all the and results, create list of quality parameters
    summary_dict = {}
    params_images = ['modeHue', 'medianSat', 'medianVal', 'avgLy', 'varLy', 'skewness', 'kurtosis', 'asg', 'sobel', 'hough', 'modaLBP', 'entropy']
    summary_dict['folder_name'] = folder_path


    # Initial checks to see the state of the folder
    print('Initial folder checks')
    f_exists, isempty, n_images, shape_images, all_images = initial_checks_func(folder_path, 5) #its the percentage of images you want

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
    #quality_images(summary_dict,join(hs,"img_quality.png"))


    # Run detection algorithm
    print('Detection algorithm')
    image_dp_dict = obtain_automatic(all_images, folder_path)


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

    """
    # Evaluate the quality of the detections
    dd=histogram_detections(summary_dict, join(hs, "hist_det.png"))
    dd.corr().style.background_gradient(cmap='coolwarm').set_precision(2)
    """

    # Run blurring algorithm
    print('Blurring algorithm')
    image_blurred_dict, folder_path_out = blur_automatic(all_images, filtered_dp_dict, folder_path)
    summary_dict['after'] = {}
    summary_dict['after']['inputs'] = {}




    # Evaluation of blurred detections
    print('Evaluate blurred detections')
    augmented_blurred_dict = check_all_blurred_quality(image_blurred_dict, all_images, folder_path_out)

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


    # Evaluation of blurred project
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
    #quality_blurred(summary_dict, join(hs, "quality_blurred.png"))


    
    # Evaluate the degradation of the whole project
    #deg_of_project(summary_dict, join(hs, "deg_proj.png"))

    # Evaluate the degradation of the images
    #degradation_images(summary_dict, join(hs, "deg_imgs.png"))

    # Evaluate the degradation of the detections !!!!
    #dm = histogram_detections_deg(summary_dict, join(hs, "deg_detections.png"))
    #dm.corr().style.background_gradient(cmap='viridis').set_precision(2)


    # Storing the quality parameters, detection and blurring performances and evaluations
    df6 = pd.DataFrame.from_dict(summary_dict) 
    df6.to_csv(store_summary_dict)        
    rec_print(summary_dict,0)


    # Create the dictionary
    report_dict = {}

    report_dict['before'] = {}
    report_dict['after'] = {}

    report_dict['before']['Cloud4D'] = {}
    report_dict['before']['Matic4D'] = {}

    report_dict['after']['Cloud4D'] = {}
    report_dict['after']['Matic4D'] = {}

    report_dict['CCompare'] = {}

    # Define the parameters that ought to be found
    params_cloud = ['n_calibrated', 'GSD', 'optim', '2D_BBA', '3D_BBA', 'keypoints_img', 'matches_img', 'mre']
    params_matic = ['dMTP1-2', 'dMTP1-3']
    params_cc = ['RMS_register', 'avg_dist', 'sigma']


    # Retrieve the results from xml files
    results_before_cloud = get_xml_thingies(report_before_path)
    print('DONE report before path')
    results_after_cloud = get_xml_thingies(report_after_path)
    print('DONE results after path')


    # Fill the 3D results dictionary
    n = 0
    for x in params_cloud:
        report_dict['before']['Cloud4D'][x] = results_before_cloud[n]
        report_dict['after']['Cloud4D'][x] = results_after_cloud[n]
        n+=1
    """
    for x in params_matic:
        report_dict['before']['Matic4D'][x] = input(x+' before:')
        report_dict['after']['Matic4D'][x] = input(x+' after:')

    for x in params_cc:
        report_dict['CCompare'][x] = input(x)
    """

    # 3D results
    def results_3d1_adapted(report_dict, name):
        aux_report = {}
        aux_report['before']={}
        aux_report['after'] = {}

        for k in report_dict['before']:
            v = report_dict['before'][k]
            for interesting in v:
                aux_report['before'][interesting] = report_dict['before'][k][interesting]
                aux_report['after'][interesting] = report_dict['after'][k][interesting]
        dx = pd.DataFrame(aux_report)

        pp =[]
        withoutnones={}
        withoutnones['before']={}
        withoutnones['after']={}
        for p in list(dx.index):
            if dx['before'][p] != None:
                if len(dx['before'][p])!=0 :
                    if dx['after'][p] != None:
                        if len(dx['after'][p])!=0:
                            pp.append(p)

    
        pdw = dx.loc[pp]
        print(pdw)
        vv = [100*(float(pdw['before'].values[i])- float(pdw['after'].values[i]) )/ float(pdw['before'].values[i]) for i in range(len(pdw['before']))]
        vv[1] = vv[1]/100
        pdw['degradation_perc'] = vv

        f, a = plt.subplots(1,2, figsize = (27, 4))
        
        m = a[0].bar(pdw.index,pdw['degradation_perc'], alpha = 0.5);

        m[0].set_facecolor('green') if pdw['degradation_perc'][0]>0 else m[0].set_facecolor('red') 
        m[1].set_facecolor('green') if pdw['degradation_perc'][1]<0 else m[1].set_facecolor('red') 
        m[2].set_facecolor('green') if pdw['degradation_perc'][2]>0 else m[2].set_facecolor('red') 
        m[3].set_facecolor('green') if pdw['degradation_perc'][3]>0 else m[3].set_facecolor('red') 
        m[4].set_facecolor('green') if pdw['degradation_perc'][4]>0 else m[4].set_facecolor('red') 
        m[5].set_facecolor('green') if pdw['degradation_perc'][5]>0 else m[5].set_facecolor('red') 
        m[6].set_facecolor('green') if pdw['degradation_perc'][6]<0 else m[6].set_facecolor('red') 
        """
        m[7].set_facecolor('blue')
        m[8].set_facecolor('blue') 
        """
        a[0].set_title('Degradation of the 3D model')
        a[0].set_ylabel('% of degradation')
        a[0].set_xlabel('Available parameters')
        a[0].legend({'Optim is /100'})
        o = -1
        for d in pdw['degradation_perc'].values:
            o = o+1
            a[0].text(o,float(d)+np.sign(d),np.round(d,3))
        """
        a[1].bar(report_dict['CCompare'].keys(), [float(f) for f in report_dict['CCompare'].values()], color='m', alpha = 0.3)
        a[1].set_xlabel('Cloud Compare parameters')
        a[1].set_title('Comparison between original and blurred 3D point clouds')
        j = -1
        for d in report_dict['CCompare'].values():
            j = j+1
            a[1].text(j,float(d)+0.03,d)
        """
        print(pdw)
        print(m)
        f.savefig(name)

    # Analyze the 3D parameters
    #results_3d1_adapted(report_dict, join(hs, "results_3d.png"))


    # Store the csv with 3D results
    df2 = pd.DataFrame.from_dict(report_dict) 
    df2.to_csv(store_report_dict)        
    rec_print(report_dict,0)
    

"""
# DO THIS -------------------------------------------------------------------------------------------------

# Provide the project folder paths
hs = "/home/asoria/Documents/proyecto_bretagne/port_kerity/"

folder_path = "/home/asoria/Documents/proyecto_bretagne/port_kerity/original_images/"
store_summary_dict = r'/home/asoria/Documents/proyecto_bretagne/port_kerity/summary_project.csv'

# UNTIL HERE ------------------------------------------------------------------------------------------------

# Fill in the paths for 3D results evaluation
report_before_path = '/home/asoria/Documents/proyecto_bretagne/port_kerity/report_original.xml'
report_after_path = '/home/asoria/Documents/proyecto_bretagne/port_kerity/report_blurred.xml'
store_report_dict = r'/home/asoria/Documents/proyecto_bretagne/port_kerity/report_project.csv'

q_analyze_blur(hs, folder_path, store_summary_dict, report_before_path, report_after_path, store_report_dict)

"""





