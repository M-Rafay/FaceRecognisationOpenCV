import cv2,os
import numpy as np
from PIL import Image
import pickle


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "Classifiers/face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
path = 'dataSet'

cam = cv2.VideoCapture(0)
font = (cv2.FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
while True:
    ret, img = cam.read()
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces= faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)

        if (nbr_predicted==60):
            nbr_predicted='unknown'
        elif (nbr_predicted == 1):
            nbr_predicted = 'Rafay'
        elif(nbr_predicted==2):
            nbr_predicted='Wasay'
        elif (nbr_predicted == 3):
            nbr_predicted = 'Iqra'
        elif (nbr_predicted == 4):
            nbr_predicted = 'Kiran'
        else:
            nbr_predicted='unknown'

        draw_text(img, str(nbr_predicted)+"--"+str(conf),x,y+h)

        cv2.imshow('im',img)
        cv2.waitKey(10)









