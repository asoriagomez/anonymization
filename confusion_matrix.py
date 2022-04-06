
from math import ceil

from numpy import rec


def are_same(rect1, rect2, threshold=0.75):
    """
    Input:
    - the detected rectangle in in format [xTL, yTL, w, h]
    - the ground truth rectangle in format [TL, BR]
    - an IoU threshold as float
    Output:
    - boolean to see if both rectangles correspond to the same object
    """
    xstart1 = rect1[0]
    xend1 = rect1[0]+rect1[2]
    ystart1 = rect1[1]
    yend1 = rect1[1]+rect1[3]

    area1 = rect1[2]*rect1[3]

    xstart2 = rect2[0][0]
    xend2 = rect2[1][0]
    ystart2 = rect2[0][1]
    yend2 = rect2[1][1]

    area2 = (xend2-xstart2)*(yend2-ystart2)

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

    unionarea = area1 + area2 - overlaparea*int(bothpositive)

    overlapping_areas = overlaparea>=threshold

    overlapping = bothpositive & overlapping_areas
    return overlapping


def confusion_matrix(ground_truth, detected_plates):
    """
    Input:
    - an array of ground truths in format [[xTL, yTL, w, h], ...]
    - an array of detected plates in format [[TL, BR], ...]
    Output:
    - a tuple of (TN, TP, FN, FP, accuracy, precision, recall, f1)
    """
    real_plates = []

    for x in range(ceil(0.5*len(ground_truth))):
        r = [ground_truth[x*2], ground_truth[x*2+1]]
        real_plates.append(r)

    TN = 0
    TP = 0

    original_plates = real_plates.copy()
    for r in original_plates:
        for d in detected_plates:
            if are_same(d, r):
                TP+=1
                real_plates.remove(r)
                detected_plates.remove(d)

    FN = len(real_plates)
    FP = len(detected_plates)

    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*precision*recall/(precision+recall)

    return (TN, TP, FN, FP, accuracy, precision, recall, f1)

# -----------------------------------------------------------------------------
# Test if algorithm works with some parameters

# detected plates in format [[xTL, yTL, w, h], ...]
detected_plates = [[663, 526, 324, 108], [26, 552, 178, 59], [1688, 235, 129, 43], [503, 587, 73, 24]]

# ground_truth in format [[TL, BR], ...]
ground_truth = [[686, 548], [987, 621], [1, 565], [186, 600], [1706, 251], [1826, 269], [1464, 316], [1555, 329]]

(TN, TP, FN, FP, accuracy, precision, recall, f1) = confusion_matrix(ground_truth, detected_plates)
print(TN, TP, FN, FP, accuracy, precision, recall, f1)

# -----------------------------------------------------------------------------------
# Precision - recall curve for various IoU thresholds
# mAP value



