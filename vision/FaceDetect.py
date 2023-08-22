import cv2

# connect to camera
capture = cv2.VideoCapture(0)
#capture = cv2.VideoCapture("markertest.mov")

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
eye_detector = cv2.CascadeClassifier("haarcascade_eye.xml") 

while True:
	ret, img = capture.read() #img type is numpy.ndarray

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # detector needs greyscale

	faces = face_detector.detectMultiScale(gray) 
	for (x, y, w, h) in faces: 
		# specifies where to draw rectangle
		# cutting rectangle starting from y to y+h and from x to x+h
		sub_gray = gray[y:y+h, x:x+h] 
		sub_color = img[y:y+h, x:x+h]
		eyes = eye_detector.detectMultiScale(sub_gray)

		# if list of eyes is nonempty, draw box on each eye
		if (len(eyes) > 0): 
			for (ex, ey, ew, eh) in eyes:
				cv2.rectangle(sub_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

		cv2.rectangle(img, (x, y), (x+w, y+h), (0,225,0), 5) 
		#print(type(img))	

	# show the image
	cv2.imshow("frame", img)

	# key 27 is esc
	if (cv2.waitKey(1) == 27):
		break

# allow other processes to use the camera after release
capture.release() 
cv2.destroyAllWindows