# -*- coding: utf-8 -*-
from cProfile import Profile
import profile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import timeit
import time
import psutil
from psutil import cpu_percent
from memory_profiler import profile
from datetime import datetime
import psutil

name = "car1"
name_open = name+".jpg"
filename = "/home/asoria/Documents/zita9999/"+name_open

#reading in the input image
plate = cv2.imread(filename)

#function that shows the image
def display(img, destination = "/home/asoria/Documents/zita9999/"+name+"_processed.png"):
    fig = plt.figure(figsize = (12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap = 'gray')
    cv2.imwrite(destination, img)

#Cascade Classifier where our hundres of samples of license plates are
plate_cascade = cv2.CascadeClassifier('/home/asoria/Documents/zita9999/haarcascades/haarcascade_russian_plate_number.xml')

@profile
def detect_plate3(img):
    
    plate_img = img.copy()
    
    

    #starttime5 = timeit.default_timer()
    starttime5  = datetime.now()
    print('psutil RAM percent before', str(psutil.virtual_memory()[2] ))
    plate_rects, rejectLevels, levelWeights  = plate_cascade.detectMultiScale3(plate_img, scaleFactor = 1.1, minNeighbors = 3, outputRejectLevels = True)	
    print('psutil RAM percent after ', str(psutil.virtual_memory()[2] ))
    #end_time5 = timeit.default_timer()
    end_time5 = datetime.now()
    diff_time5 = end_time5 - starttime5
    diff_time5 = diff_time5.total_seconds() #for the datetime.now() function

    #draws the rectangle around it
    i=0
    for (x,y,w,h) in plate_rects:
        i=i+1
        cv2.rectangle(plate_img, (x,y), (x+w, y+h), (255,0,0), 5)
        a=int(y+h/2)
        cv2.putText(plate_img,str(i),(x,a), cv2.FONT_ITALIC, 0.9,(0,0,255),2,cv2.LINE_AA)
    print(plate_rects) 
    return plate_img, rejectLevels, levelWeights, diff_time5, plate_rects

result5, rejectLevels, levelWeights5, diff_time5,  plate_rects5 = detect_plate3(plate)

print("levelWeights for confidence = ", levelWeights5)
print('rejectLevels for Trainig Steps = ', rejectLevels)
print("The time difference for detectMultiScale3 is :", diff_time5)
display(result5, destination = "/home/asoria/Documents/zita9999/"+name+"_levelW.png")


"""  Blurs the license plate
@profile
def detect_blur(img, plate_rects):
    
    plate_img = img.copy()
        
    for (x,y,w,h) in plate_rects:
        x_offset = x
        y_offset = y
        
        x_end = x+w
        y_end = y+h
        
        zoom_img = plate_img[y_offset:y_end, x_offset:x_end]

        #starttime_blur = timeit.default_timer()
        starttime_blur = process_time() 
        cpu_usage = cpu_percent(0)
        zoom_img = cv2.medianBlur(zoom_img,15)
        #delay_blur = timeit.default_timer() - starttime_blur
        delay_blur = process_time() - starttime_blur
        


        plate_img[y_offset:y_end, x_offset:x_end] = zoom_img
        
        
    return plate_img, delay_blur, cpu_usage

result4, delay_blur, cpu_usage = detect_blur(plate, plate_rects5)
display(result4, destination = "/home/asoria/Documents/zita9999/"+name+"_blurredX.png")
print("The time difference for detectBlur is :", delay_blur)
#print("The cpu usage for detectBlur is :", cpu_usage)
"""








