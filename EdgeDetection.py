
import cv2
import numpy as np

#image = cv2.imread('Penguins.jpg')
#Canny () function for detecting the edges of the already read image.
#cv2.imwrite('edges_Penguins.png',cv2.Canny(image,200,300))
#cv2.imshow('edges',cv2.imread('edges_Penguins.png'))


#Use HaarCascadeClassifier
f#ace_detection = cv2.CascadeClassifier('ml lab 8/haarcascades/haarcascades/haarcascade_frontalface_default.xml')

#img2 = cv2.imread('ml lab 8/Lab8Images/AB.jpg')

#gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#faces = face_detection.detectMultiScale(gray, 1.3, 5)

#for (x,y,w,h) in faces:
 #img2 = cv2.rectangle(img2,(x,y),(x+w, y+h),(0,255,0),10)
#cv2.imwrite('Face_AB.jpg',img2)

#cv2.imshow('Face_AB.jpg',img2)

eye_cascade = cv2.CascadeClassifier('/haarcascades/haarcascade_eye.xml')
img = cv2.imread('AB_Eye.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eyes = eye_cascade.detectMultiScale(gray, 1.03, 5)


for (ex,ey,ew,eh) in eyes:
    img = cv2.rectangle(img,(ex,ey),(ex+ew, ey+eh),(0,255,0),20)
cv2.imwrite('Eye_AB.jpg',img)

cv2.imshow('Eye_AB.jpg', img)
