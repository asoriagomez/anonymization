# -*- coding: utf-8 -*-
from cProfile import Profile
import profile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import timeit
from time import process_time
import psutil
from psutil import cpu_percent
from memory_profiler import profile



print('memory usage = ',psutil.Process().memory_info().rss / (1024 * 1024),'MBytes')

name = "car1"
name_open = name+".jpg"

#reading in the input image
plate = cv2.imread("/home/asoria/Documents/zita9999/"+name_open)


#function that shows the image
def display(img, destination = "/home/asoria/Documents/zita9999/"+name+"_processed.png"):
    fig = plt.figure(figsize = (12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap = 'gray')
    cv2.imwrite(destination, img)

""""
#need to change color of picture from BGR to RGB
plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
#display(plate, destination = "/home/asoria/Documents/zita9999/"+name+"_rgb.png")
"""

#Cascade Classifier where our hundres of samples of license plates are
plate_cascade = cv2.CascadeClassifier('/home/asoria/Documents/zita9999/haarcascades/haarcascade_russian_plate_number.xml')

""""
def detect_plate(img):
    
    plate_img = plate.copy()
    
    #gets the points of where the classifier detects a plate
    #YOU ARE MODIFYING THIS LINE
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.1, minNeighbors = 2)

    #draws the rectangle around it
    for (x,y,w,h) in plate_rects:
        cv2.rectangle(plate_img, (x,y), (x+w, y+h), (255,0,0), 5)

    return plate_img

result2 = detect_plate(plate)
#display(result2, destination = "/home/asoria/Documents/zita9999/"+name+"_scale2_1_neig2.png")
"""

#@profile
def detect_plate3(img):
    
    plate_img = plate.copy()
    
    starttime5 = timeit.default_timer()
    #starttime5  = process_time() 

    plate_rects, rejectLevels, levelWeights  = plate_cascade.detectMultiScale3(plate_img, scaleFactor = 1.3, minNeighbors = 3, \
        outputRejectLevels = True)	

    diff_time5 = timeit.default_timer() - starttime5
    #diff_time5 = process_time() - starttime5
    
    #draws the rectangle around it
    i=0
    for (x,y,w,h) in plate_rects:
        i=i+1
        cv2.rectangle(plate_img, (x,y), (x+w, y+h), (255,0,0), 5)
        a=int(y+h/2)
        cv2.putText(plate_img,str(i),(x,a), cv2.FONT_ITALIC, 0.9,(0,0,255),2,cv2.LINE_AA)
    
    return plate_img, rejectLevels, levelWeights, diff_time5, plate_rects

result5, rejectLevels, levelWeights5, diff_time5,  plate_rects5 = detect_plate3(plate)

print("levelWeights for confidence = ", levelWeights5)
print('rejectLevels for Trainig Steps = ', rejectLevels)
print("The time difference for detectMultiScale3 is :", diff_time5)
display(result5, destination = "/home/asoria/Documents/zita9999/"+name+"_levelW.png")


"""
#detects the plate and zooms in on it
def detect_zoom_plate(img, kernel):
    
    plate_img = img.copy()
    
    #gets the points of where the classifier detects a plate
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor =1.9, minNeighbors = 2) #maxSize = (100,100))
    
    for (x,y,w,h) in plate_rects:
        x_offset = x
        y_offset = y
        
        x_end = x+w
        y_end = y+h
        
        #getting the points that show the license plate
        zoom_img = plate_img[y_offset:y_end, x_offset:x_end]
        #increasing the size of the image
        zoom_img = cv2.resize(zoom_img, (0,0),fx = 2, fy = 2)
        zoom_img = zoom_img[7:-7, 7:-7]
        #sharpening the image to make it look clearer
        zoom_img = cv2.filter2D(zoom_img, -1, kernel)
        
        zy = (40 - (y_end - y_offset))//2
        zx = (144 - (x_end-x_offset))//2
        
        ydim = (y_end+zy-50) - (y_offset-zy-50)
        xdim = (x_end+zx) - (x_offset-zx)
       
       
        zoom_img = cv2.resize(zoom_img,(xdim,ydim))
        
        #putting the zoomed in image above where the license plate is located
        try:
            plate_img[y_offset-zy-55:y_end+zy-55, x_offset-zx:x_end+zx] = zoom_img
        except:
            pass
         
        #drawing a rectangle
        for (x,y,w,h) in plate_rects:
            cv2.rectangle(plate_img, (x,y), (x+w, y+h), (255,0,0), 2)
            
        
    return plate_img
"""

#blurs the license plate
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

""""
#matrix needed to sharpen the image
kernel = np.array([[-1,-1,-1],
                   [-1,9,-1],
                   [-1,-1,-1]])
    
#result3 = detect_zoom_plate(plate, kernel)
#display(result3, destination = "/home/asoria/Documents/zita9999/"+name+"_zoomed.png")
"""

result4, delay_blur, cpu_usage = detect_blur(plate, plate_rects5)
display(result4, destination = "/home/asoria/Documents/zita9999/"+name+"_blurred.png")
print("The time difference for detectBlur is :", delay_blur)
#print("The cpu usage for detectBlur is :", cpu_usage)



"""
#### video
cap = cv2.VideoCapture('C:/Users/chris/OneDrive/Documents/Apowersoft/Video Editor Pro/Output/MyVideo_5.mp4')

#gets the height and width of each frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#saves the video to a file
writer = cv2.VideoWriter('C:/Users/chris/OneDrive/Documents/Courses and Projects/Computer Vision/vid_zoom2.mp4', cv2.VideoWriter_fourcc(*'DIVX'),20,(width, height))

#to make video actual speed
if cap.isOpened() == False:
    print('error file not found')

#while the video is running the loop will keep running
while cap.isOpened():
    #returns each frame
    ret, frame = cap.read()
    
    # if there are still frames keeping showing the video
    if ret == True:
        #apply our detect and zoom function to each frame
        frame = detect_zoom_plate(frame, kernel)
        #show the frame
        cv2.imshow('frame', frame)
        writer.write(frame)
        
        #will stop the video if it fnished or you press q
        if cv2.waitKey(10) & 0xff == ord('q'):
            break
    else:
        break

#stop the video, and gets rid of the window that it opens up        
cap.release()
writer.release()
cv2.destroyALLWindows()
"""








