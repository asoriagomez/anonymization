from statistics import median
from tkinter import N
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from os.path import isfile, join



# Quality of the images
def quality_images(summary_dict, name):

    qu_params = ['modeHue','medianSat', 'medianVal', 'avgLy', 'varLy', 'skewness', 'kurtosis', 'asg', 'sobel', 'hough', 'modaLBP', 'entropy']
    qu_params
    info = {}
    for q in qu_params:
        info[q]=[]

    for img in summary_dict['before']['inputs']['images']:
        v = summary_dict['before']['inputs']['images'][img]['img_char']
        for q in qu_params:
            a = summary_dict['before']['inputs']['images'][img]['img_char'][q]
            
            info[q].append(a)

    dd = pd.DataFrame(info)

    qu_params = ['modeHue','medianSat', 'medianVal', 'avgLy', 'varLy', 'skewness', 'kurtosis', 'asg', 'sobel', 'hough', 'modaLBP', 'entropy']

    row_old = 0
    col = 0
    colors = ['b', 'k', 'b', 'k', 'b', 'k', 'b', 'k', 'b', 'k', 'b', 'k']
    f,a = plt.subplots(4, 3, figsize = (23,15))

    for i in range(12):
        p = qu_params[i]

        row = int(np.floor(i/3))
        if row_old!=row:
            col=0
        
        row_old = row
        values = list(dd[p])
        a[row][col].hist(values,align='left', bins = 30, color = colors[i], alpha = 0.7, label = ['Avg = '+ str(np.round(np.mean(values),2))+ '; Var = '+ str(np.round(np.var(values),2))])
        a[row][col].legend()
        a[row][col].set_xlabel(p)
        a[row][col].grid()
        col = col+1
    plt.suptitle('Histograms of quality of **images**')
    f.savefig(name)

# Quality of the blurred
def quality_blurred(summary_dict, name):

    qu_params = ['modeHue','medianSat', 'medianVal', 'avgLy', 'varLy', 'skewness', 'kurtosis', 'asg', 'sobel', 'hough', 'modaLBP', 'entropy']
    qu_params
    info = {}
    for q in qu_params:
        info[q]=[]

    for img in summary_dict['after']['inputs']['images']:
        v = summary_dict['after']['inputs']['images'][img]['img_char']
        for q in qu_params:
            a = summary_dict['after']['inputs']['images'][img]['img_char'][q]
            
            info[q].append(a)

    dd = pd.DataFrame(info)

    qu_params = ['modeHue','medianSat', 'medianVal', 'avgLy', 'varLy', 'skewness', 'kurtosis', 'asg', 'sobel', 'hough', 'modaLBP', 'entropy']

    row_old = 0
    col = 0
    colors = ['m', 'k', 'm', 'k', 'm', 'k', 'm', 'k', 'm', 'k', 'm', 'k']
    f,a = plt.subplots(4, 3, figsize = (23,15))

    for i in range(12):
        p = qu_params[i]

        row = int(np.floor(i/3))
        if row_old!=row:
            col=0
        
        row_old = row
        values = list(dd[p])
        a[row][col].hist(values,align='left', bins = 30, color = colors[i], alpha = 0.7, label = ['Avg = '+ str(np.round(np.mean(values),2))+ '; Var = '+ str(np.round(np.var(values),2))])
        a[row][col].legend()
        a[row][col].set_xlabel(p)
        a[row][col].grid()
        col = col+1
    plt.suptitle('Histograms of quality of **blurred**')
    f.savefig(name)

# 1) Histogram of the number of detections per image
def ndet_peri(summary_dict, name):

    nd_img = {}
    for k in summary_dict['before']['inputs']['images']:
        v = summary_dict['before']['inputs']['images'][k]
        nd = len(list(v['detections'].keys()))
        nd_img[k] = nd
    nd_img_list = list(nd_img.values())
    fig, ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(10)
    ax.set_ylabel('N° repetitions')
    ax.set_xlabel('Detections per image')
    ax.set_title('Histogram of number of detections per image')
    mm = np.max(nd_img_list)+2
    n, bins, patches = ax.hist(nd_img_list, color = 'm', bins = range(mm),align='left')
    xticks = [(bins[idx+1] + value)/2 for idx, value in enumerate(bins[:-1])]
    for idx, value in enumerate(n):
        plt.text(xticks[idx]-0.5, value+1, int(value), ha='center')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(range(mm))
    plt.grid()
    fig.savefig(name)
    plt.show()



# Histogram of the size of detections
def hist_size_det(summary_dict, name, hs):
    size_det = []
    for k in summary_dict['before']['inputs']['images']:
        v = summary_dict['before']['inputs']['images'][k]['detections']
        image = cv2.imread(join(hs,k))
        s = image.shape
        for k1 in v:
            coo = summary_dict['before']['inputs']['images'][k]['detections'][k1]['coordis']
            siz = 100*coo[2]*coo[3] /(s[0]*s[1])
            size_det.append(siz)
    fig1, ax1 = plt.subplots()
    fig1.set_figheight(5)
    fig1.set_figwidth(10)
    ax1.set_ylabel('N° repetitions')
    ax1.set_xlabel('Percentage of the size of detections')
    ax1.set_title('Histogram of the  size of detections')
    n1, bins1, patches1 = ax1.hist(size_det, color = 'g', bins = 50 , align='left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid()
    fig1.savefig(name)
    plt.show()

# Quality of detections
def histogram_detections(summary_dict, name):
    qu_params = ['modeHue','medianSat', 'medianVal', 'avgLy', 'varLy', 'skewness', 'kurtosis', 'asg', 'sobel', 'hough', 'modaLBP', 'entropy']
    qu_params
    infor = {}
    for q in qu_params:
        infor[q]=[]
    for img in summary_dict['before']['inputs']['images']:
        v = summary_dict['before']['inputs']['images'][img]['detections']
        for d in v.keys():
            cosas = v[d]
            for q in qu_params:
                infor[q].append(cosas[q])

    dd = pd.DataFrame(infor)

    qu_params = ['modeHue','medianSat', 'medianVal', 'avgLy', 'varLy', 'skewness', 'kurtosis', 'asg', 'sobel', 'hough', 'modaLBP', 'entropy']

    row_old = 0
    col = 0
    colors = ['b', 'k', 'gray', 'y', 'y', 'c', 'c', 'g', 'k', 'r', 'orange', 'magenta']
    f,a = plt.subplots(4, 3, figsize = (23,15))

    for i in range(12):
        p = qu_params[i]

        row = int(np.floor(i/3))
        if row_old!=row:
            col=0
        
        row_old = row
        values = list(dd[p])
        a[row][col].hist(values, align='left',bins = 30, color = colors[i], alpha = 0.55, label = ['Avg = '+ str(np.round(np.mean(values),2))+ '; Var = '+ str(np.round(np.var(values),2))])
        a[row][col].legend()
        a[row][col].set_title(p)
        a[row][col].grid()
        col = col+1
    plt.suptitle('Histograms of quality of detections')
    f.savefig(name)
    return dd
"""
It's because there's no variance/ standard deviation in the second column 
and thus in the correlation coefficient calculation when you divide by std or var 
(however it's implemented) you're in turn dividing zero by zero which yield nan.
"""
#dd.corr().style.background_gradient(cmap='coolwarm').set_precision(2)



# Evaluation of the detection algorithm
def hist_eval_det(image_f1_dict, name):
    params = ['TP', 'FN', 'FP', 'recall', 'precision', 'f1']
    infom = {}
    for p in params:
        infom[p]=[]
    for img in image_f1_dict:
        for p in params:
            infom[p].append(image_f1_dict[img][p])

    dd = pd.DataFrame(infom)


    row_old = 0
    col = 0
    colors = ['b', 'y', 'c', 'g', 'r', 'orange', 'magenta']
    f,a = plt.subplots(2, 3, figsize = (16,10))

    for i in range(6):
        p = params[i]

        row = int(np.floor(i/3))
        if row_old!=row:
            col=0
        
        row_old = row
        values = list(dd[p])
        a[row][col].hist(values, align='left',bins = 30, color = colors[i], alpha = 0.6, label = ['Avg = '+ str(np.round(np.mean(values),2))+ '; Var = '+ str(np.round(np.var(values),2))])
        a[row][col].legend()
        a[row][col].set_title(p)
        a[row][col].grid()
        col = col+1
    plt.suptitle('Histograms of evaluation of detection algorithm')
    f.savefig(name)



def deg_of_project(summary_dict, name):
    pars = list(summary_dict['before']['inputs'].keys())
    pars.remove('images')
    deltas = []
    for p in pars:
        delta = 100%(summary_dict['before']['inputs'][p] - summary_dict['after']['inputs'][p])/summary_dict['before']['inputs'][p]
        deltas.append(delta)
    fig1, ax1 = plt.subplots()
    fig1.set_figheight(5)
    fig1.set_figwidth(10)
    ax1.set_ylabel('% of degradation')
    ax1.set_xlabel('Parameters')
    ax1.bar(pars, deltas,color='k', alpha = 0.45)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    for i in range(len(pars)):
        plt.text(i, deltas[i]+np.sign(deltas[i])*0.0001, np.round(deltas[i],3), ha='center')
    ax1.set_title('Degradation of a project: (before-after)/before')
    fig1.savefig(name)
    plt.show()




# Histogram of percentages of degradation of each parameter of the images
def degradation_images(summary_dict, name, info):

    qu_params = ['modeHue','medianSat', 'medianVal', 'avgLy', 'varLy', 'skewness', 'kurtosis', 'asg', 'sobel', 'hough', 'modaLBP', 'entropy']
    qu_params
    info = {}
    for q in qu_params:
        info[q]=[]

    for img in summary_dict['before']['inputs']['images']:
        for q in qu_params:
            diff = summary_dict['before']['inputs']['images'][img]['img_char'][q]-summary_dict['after']['inputs']['images'][img]['img_char'][q]
            perc = 100*diff/(summary_dict['before']['inputs']['images'][img]['img_char'][q])
            info[q].append(perc)

    dd = pd.DataFrame(info)

    qu_params = ['modeHue','medianSat', 'medianVal', 'avgLy', 'varLy', 'skewness', 'kurtosis', 'asg', 'sobel', 'hough', 'modaLBP', 'entropy']

    row_old = 0
    col = 0
    colors = ['b', 'k', 'gray', 'y', 'y', 'c', 'c', 'g', 'k', 'r', 'orange', 'magenta']
    f,a = plt.subplots(4, 3, figsize = (23,15)) if info else None
    medians_images = []
    for i in range(12):
        p = qu_params[i]

        row = int(np.floor(i/3))
        if row_old!=row:
            col=0
        
        row_old = row
        values = list(dd[p])
        if info:
            a[row][col].hist(values,align='left', bins = 30, color = colors[i], alpha = 0.7, label = ['Avg = '+ str(np.round(np.mean(values),2))+ '; Var = '+ str(np.round(np.var(values),2))])
            a[row][col].legend()
            a[row][col].set_xlabel("% "+ p)
            a[row][col].grid()
        else:
            None
        col = col+1
        medians_images.append(np.median(values))
    plt.suptitle('Histograms of relative degradation of **images**')
    f.savefig(name)

    return medians_images


def histogram_detections_deg(summary_dict, name, info):
    qu_params = ['modeHue','medianSat', 'medianVal', 'avgLy', 'varLy', 'skewness', 'kurtosis', 'asg', 'sobel', 'hough', 'modaLBP', 'entropy']
    qu_params
    infor = {}
    for q in qu_params:
        infor[q]=[]
    for img in summary_dict['before']['inputs']['images']:
        v = summary_dict['before']['inputs']['images'][img]['detections']
        z = summary_dict['after']['inputs']['images'][img]['detections']
        for d in v.keys():
            cosas = v[d]
            cosas_after = z[d]
            for q in qu_params:
                infor[q].append(100*(cosas[q]-cosas_after[q])/(cosas[q]+0.0000000000001))

    dd = pd.DataFrame(infor)

    qu_params = ['modeHue','medianSat', 'medianVal', 'avgLy', 'varLy', 'skewness', 'kurtosis', 'asg', 'sobel', 'hough', 'modaLBP', 'entropy']

    row_old = 0
    col = 0
    colors = ['b', 'k', 'gray', 'y', 'y', 'c', 'c', 'g', 'k', 'r', 'orange', 'magenta']
    f,a = plt.subplots(4, 3, figsize = (23,17)) if info else None
    medians_detections = []
    for i in range(12):
        p = qu_params[i]

        row = int(np.floor(i/3))
        if row_old!=row:
            col=0
        
        row_old = row
        values = list(dd[p])
        if info:
            a[row][col].hist(values, align='left',bins = 30, color = colors[i], alpha = 0.99, label = ['Avg = '+p+ " "+str(np.round(np.mean(values),2))+ '; Var = '+ str(np.round(np.var(values),2))])
            a[row][col].legend()
            a[row][col].set_xlabel('% '+p)
            a[row][col].grid()
        else:
            None
        col = col+1
        medians_detections.append(np.median(values))
    plt.suptitle('Histograms of relative degradation (%) of **detections**')
    f.savefig(name)
    return dd, medians_detections
#dd.corr().style.background_gradient(cmap='viridis').set_precision(2)






import math
from statsmodels.stats import diagnostic
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

def f1_pp(summary_dict):
    f1_predict = {}
    for img in summary_dict['before']['inputs']['images']:
        f1_predict[img] = {}
        f1 = summary_dict['before']['inputs']['images'][img]['F1']
        f1_predict[img]['F1'] = f1
        for p in summary_dict['before']['inputs']['images'][img]['img_char']:
            f1_predict[img][p] = summary_dict['before']['inputs']['images'][img]['img_char'][p]

    df_f1 = pd.DataFrame(f1_predict)
    df_f1t = df_f1.transpose()
    #df_f1t.sample(5)
    mod = smf.ols(formula='F1 ~ modeHue + medianSat + medianVal + avgLy + varLy + skewness + kurtosis + asg + sobel + hough + modaLBP + entropy', data=df_f1t)
    np.random.seed(2)
    res = mod.fit()
    print(res.summary())