


import cv2
import matplotlib.pyplot as plt
image = cv2.imread("/home/asoria/Documents/proyecto_bretagne/port_kerity/original_images/Image_000050.jpg")
img = image.copy()


cv2.rectangle(img, (187,642), (428, 755), (255,0,0), 5)
cv2.putText(img,'Ground truth',(187,635), cv2.FONT_ITALIC, 0.9,(255,0,0),2,cv2.LINE_AA)

cv2.rectangle(img, (115,585), (330, 855), (0,255,0), 5)
cv2.putText(img,'Detection',(115,580), cv2.FONT_ITALIC, 0.9,(0,255,0),2,cv2.LINE_AA)

fig = plt.figure(figsize = (12,10))
ax = fig.add_subplot(111)
show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
ax.imshow(show,cmap = 'gray')
plt.title('IoU example')
plt.show()