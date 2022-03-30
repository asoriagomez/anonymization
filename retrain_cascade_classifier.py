import cv2
import matplotlib.pyplot as plt
print(cv2.__version__)


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


#Cascade Classifier where our hundres of samples of license plates are
plate_cascade = cv2.CascadeClassifier('/home/asoria/Documents/zita9999/haarcascades/haarcascade_russian_plate_number.xml')
