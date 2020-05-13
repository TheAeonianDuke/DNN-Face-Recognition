import numpy as np
import cv2


def detect(cascade):
	cam=cv2.VideoCapture(0)
	while True:
		ret,img=cam.read()
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert image to gray
		faces=cascade.detectMultiScale(gray,1.3,5) #Detect objects of different sizes 

		for (x,y,w,h) in faces:
			img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]

		cv2.imshow('Face',img)
		cv2.waitKey(1)
	cv2.destroyAllWindows()

cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')
detect(cascade)