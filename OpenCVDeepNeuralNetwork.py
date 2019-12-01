import numpy as np 					# Linear algebra
import pandas as pd 				# Data processing
import matplotlib.pyplot as plt 	# Visiulazation
import cv2 							# Image processing
import os 							# Systems

modelFile ="res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

#function to extract box dimensions
def face_dnn(img, coord=False):
    blob = cv2.dnn.blobFromImage(img, 1, (224,224), [104, 117, 123], False, False) #
    # params: source, scale=1, size=300,300, mean RGB values (r,g,b), rgb swapping=false, crop = false
    conf_threshold = 0.8 			# Confidence at least 80%
    frameWidth=img.shape[1] 		# Get image width
    frameHeight=img.shape[0] 		# Get image height
    max_confidence=0
    net.setInput(blob)
    detections = net.forward()
    detection_index=0
    bboxes = []
    count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            
            if max_confidence < confidence: # Only show maximum confidence face
                max_confidence = confidence
                detection_index = i
                count += 1
    i=detection_index        
    x1 = int(detections[0, 0, i, 3] * frameWidth)
    y1 = int(detections[0, 0, i, 4] * frameHeight)
    x2 = int(detections[0, 0, i, 5] * frameWidth)
    y2 = int(detections[0, 0, i, 6] * frameHeight)
    cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),2)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if coord==True:
        return x1, y1, x2, y2
    print('Number of faces detected:', count)
    return cv_rgb

def nfaces_dnn(img):
    blob = cv2.dnn.blobFromImage(img, 1.2, (1200,1200), [104, 117, 123], False, False) #
    # params: source, scale=1, size=300,300, mean RGB values (r,g,b), rgb swapping=false, crop = false
    conf_threshold= 0.8         	# Confidence at least 80%
    frameWidth=img.shape[1] 		# Get image width
    frameHeight=img.shape[0] 		# Get image height
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    count = 0
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),2)
            count += 1
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print('Number of faces detected:', count)
    return cv_rgb

print("Number of faces: 1")
celeb = cv2.imread("obama.jpg")
a=face_dnn(celeb)
plt.imshow(a)
plt.show()
print("Accuracy of the prediction: 100%")

print()

print("Number of faces: 7")
img=cv2.imread("people.png")
c=nfaces_dnn(img)
plt.figure(figsize=(15,18))
plt.imshow(c)
plt.show()
print("Accuracy of the prediction: 85.71%")

print()

"""

OpenCV Deep Neural Networks

Observations: 

This model was able to detect more correct faces and detect non-frontal faces
without clipping chincompared to the haar cascade method. However, for the group picture
the model didn't register some of the faces of people

"""