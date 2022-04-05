import numpy as np
from operator import itemgetter
import collections
import cv2
import matplotlib.pyplot as plt

def are_overlapping(rect1, rect2):
    """
    Auxiliar function to see if two rectangles are overlapping:
    - Have a positive area of overlap
    - Be greater than a threshold defined as 0.5 of the maximum possible overlap area (one inside the other)
    """

    rect1 = [ 450 , 421 ,1050 , 350]
    rect2 = [ 663 , 526 , 324 , 108]

    xstart1 = rect1[0]
    xend1 = rect1[0]+rect1[2]
    ystart1 = rect1[1]
    yend1 = rect1[1]+rect1[3]

    area1 = rect1[2]*rect1[3]

    xstart2 = rect2[0]
    xend2 = rect2[0]+rect2[2]
    ystart2 = rect2[1]
    yend2 = rect2[1]+rect2[3]

    area2 = rect2[2]*rect2[3]

    xstartmin = min(xstart1, xstart2)
    xstartmax = max(xstart1, xstart2)
    xendmin = min(xend1, xend2)
    xendmax = max(xend1, xend2)

    ystartmin = min(ystart1, ystart2)
    ystartmax = max(ystart1, ystart2)
    yendmin = min(yend1, yend2)
    yendmax = max(yend1, yend2)

    xoverlap = xendmin - xstartmax
    yoverlap = yendmin - ystartmax

    bothpositive = (xoverlap>0) and (yoverlap>0)

    overlaparea = xoverlap*yoverlap
    threshold = 0.5*min(area1, area2)/max(area1, area2)
    overlapping_areas = overlaparea>=threshold

    overlapping = bothpositive and overlapping_areas
    return overlapping


def display(img, keep, destination = "/home/asoria/Documents/zita9999/nms.png"):
    """
    Auxiliar method to display an image
    """
    i=0
    for (x,y,w,h) in keep:
        i=i+1
        cv2.rectangle(plate, (x,y), (x+w, y+h), (255,0,0), 5)
        a=int(y+h/2)
        cv2.putText(plate,str(i),(x,a), cv2.FONT_ITALIC, 0.9,(0,0,255),2,cv2.LINE_AA)
    fig = plt.figure(figsize = (12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap = 'gray')
    cv2.imwrite(destination, img)


def NMS(plate_rects, levelWeights):
    """
    Inputs:
    - all detections as rectangles
    - confidence weights for all detections
    Outputs:
    - filters duplicate detections
    """
    both_lists = list(zip(levelWeights, plate_rects))
    sorted_lists = sorted(both_lists, reverse = True, key=lambda x: x[0])

    rects = [x[1] for x in sorted_lists]

    keep = []

    keep.append(rects[0])
    rects.remove(rects[0])
    while len(rects)>0:
        ref_rect = keep[-1]
        for r in rects:
            if are_overlapping(r, ref_rect):
                rects.remove(r)
        if len(rects)>0:
            keep.append(rects[0])

    print(keep)
    return keep

# ----------------------------------------------------------------------------------------------------------
# Trying the algorithms with some parameters

name = "car1"
name_open = name+".jpg"
filename = "/home/asoria/Documents/zita9999/"+name_open

plate = cv2.imread(filename)

plate_rects = [[1688 , 235 , 129  , 43],
 [1711 , 233 , 158  , 53],
 [ 450 , 421 , 1050 , 350],
 [ 503 , 587  , 73 ,  24],
 [ 663 , 526 , 324 , 108],
 [  26 , 552 , 178 ,  59]]

levelWeights =  [1.46363363,
0.51568082,
1.35041222,
0.62210676,
3.3027389,
3.3027389]

keep = NMS(plate_rects, levelWeights)

display(plate, keep)