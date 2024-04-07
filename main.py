import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



#MOBILE NET SSD MODEL

img_path = 'assets/princessgardens_5967.0.jpg'
prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
model_path ='models/MobileNetSSD_deploy.caffemodel'
min_confidence = 0.2

classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

colors = np.random.uniform(0,255,size=(len(classes),3))
np.random.seed(543210)
net=cv.dnn.readNetFromCaffe(prototxt_path,model_path)
image = cv.imread(img_path)
height,width = image.shape[0],image.shape[1]
blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 0.007, (300, 300), 130)
net.setInput(blob)
detected_objects = net.forward()
print(detected_objects[0][0][0])
for i in range (detected_objects.shape[2]):
    confidence = detected_objects[0][0][i][2]
    if confidence > min_confidence:
        class_index =int(detected_objects[0,0,i,1])
        upper_left_x = int(detected_objects[0,0,i,3] * width)
        upper_left_y = int(detected_objects[0,0,i,4] * height)
        lower_right_x = int(detected_objects[0,0,i,5] * width)
        lower_right_y = int(detected_objects[0,0,i,6] * height)

        prediction_text = f"{classes[class_index]}:{confidence:.2f}%"
        cv.rectangle(image,(upper_left_x,upper_left_y),(lower_right_x,lower_right_y),colors[class_index],3)
        cv.putText(image,prediction_text,(upper_left_x,upper_left_y - 15 if upper_left_y > 30 else upper_left_y + 15),cv.FONT_HERSHEY_SIMPLEX,0.6,colors[class_index],2)

cv.imshow("Detect Objects",image)
cv.waitKey(0)
cv.destroyAllWindows
'''
img = cv.imread('assets/tomato.jpeg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

   
# Creates the environment 
# of the picture and shows it
plt.subplot(1, 1, 1)
plt.imshow(img_gray)
plt.show()
'''