
import cv2
import numpy as np

eye_cascade = cv2.CascadeClassifier('/haarcascades/haarcascade_eye.xml')
img = cv2.imread('AB_Eye.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eyes = eye_cascade.detectMultiScale(gray, 1.03, 5)


for (ex,ey,ew,eh) in eyes:
    img = cv2.rectangle(img,(ex,ey),(ex+ew, ey+eh),(0,255,0),20)
cv2.imwrite('Eye_AB.jpg',img)

cv2.imshow('Eye_AB.jpg', img)
