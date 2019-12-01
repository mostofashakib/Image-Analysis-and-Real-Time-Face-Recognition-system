import numpy as np 							# Linear Algebra
import pandas as pd 						# Data Processing
import matplotlib.pyplot as plt 			# Visiulazation
import cv2 									# Image Processing
import os 									# Systems
import numpy as np                          # Array Manipulation
import cv2								    # Real-time open source computer vision library

joe_image   =  "joe.jpg"
joe_imread  =   cv2.imread(joe_image)

obama_image   =  "obama.jpg"
obama  =   cv2.imread(obama_image)

def show_image(image):
    plt.figure(figsize=(8,5))
    #Before showing image, bgr color order transformed to rgb order
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()

def face_detection(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    print('Number of faces detected:', len(faces))

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv_rgb

print("Number of faces: 1")
joe_pic = face_detection(joe_imread)
plt.imshow(joe_pic)
plt.show()
print("Accuracy of the prediction: 100%",)

print()

print("Number of face: 1")
obama_pic = face_detection(obama)
plt.imshow(obama_pic)
plt.show()
print("Accuracy of the prediction: 50%")

print()

print("Number of faces: 7")
plt.figure(figsize = (15,18))
Group_picture = cv2.imread("people.png")
Group_Photo = face_detection(Group_picture)
plt.imshow(Group_Photo)
plt.show()
print("Accuracy of the prediction: 54%")

print()

"""
Haar Cascade is a massive XML file developed by Intel with a lot of feature sets and this feature set corresponds
to specific type of objects.

Observation 1: 

W can conclude that face detection using haar cascade is accurate for pictures with less
facial pictures, whoever, when the number of facial features in a picture increases then the method is not accurate.

Observation 2:

We can also observe that for pictures with more than more than one people the method is not accurate.

Observation 3:

The Haar Cascade method detects non-frontal face images and also boxes sometimes do not include
full face, clipping chins or forehead
"""