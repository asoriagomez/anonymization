from confusion_matrix_calculation import *
from license_plate import *
from get_pixel_coordinates import *
from nms_algorithm import *

# Necessary parameters ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

name = "car1"
name_open = name+".jpg"
filename = "/home/asoria/Documents/zita9999/"+name_open

plate = cv2.imread(filename)
scaleF = 1.1
minNei = 3

# Find ground truth ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# to not repeat the process all the time
ground_truth = [[686, 548], [987, 621], [1, 565], [186, 600], [1706, 251], [1826, 269]]
print('Press any key to stop the algorithm and dont close the RGB pixels car:')
print('ground truth = ', ground_truth)
#ground_truth = get_coordinates(filename)


scaleFactors = np.linspace(1.1, 20, 50)
minNeighbours = range(1,10,1)
acs = []
ps = []
recalls = []
f1s = []

for m in minNeighbours:
        
    # Detect license plates ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    img_detected, _, levelWeights, _,  detected_plates = detect_plate3(plate, scaleF, m)
    #display(img_detected, destination = "/home/asoria/Documents/zita9999/"+name+"_detected_sF_"+str(s)+"_nei_"+str(minNei)+".png", title = 'detected: sF='+str(s)+", minNei="+str(minNei))
    # Remove duplicates -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    keep = NMS(detected_plates, levelWeights)
    display(plate,"/home/asoria/Documents/zita9999/"+name+"_cleaned_sF_"+str(scaleF)+"_nei_"+str(m)+".png",'cleaned: sF='+str(scaleF)+", minNei="+str(m),keep)

    # Calculate confusion matrix --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    (TN, TP, FN, FP, a, p, r, f1) = confusion_matrix_calc(ground_truth, keep, 0.55)
    acs.append(a)
    ps.append(p)
    recalls.append(r)
    f1s.append(f1)

   


mAP = auc(ps, recalls)
print('mAP=', mAP)

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.plot(minNeighbours,acs, 'go-')
plt.title('Accuracies vs minNeighbours')
plt.ylabel('Accuracy')

plt.subplot(2,2,3)
plt.plot(minNeighbours,f1s, 'mo-')
plt.title('F1s vs minNeighbours')
plt.ylabel('F1')
plt.xlabel('minNeighbours')

plt.subplot(2,2,(2,4))
plt.plot(ps,recalls, 'bo-')
plt.title('Precision vs recall')
plt.xlabel('Recall or TPR')
plt.ylabel('Precision')

plt.show()



