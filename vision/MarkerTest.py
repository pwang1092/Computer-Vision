import cv2

# Load the dictionary that was used to generate the markers
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# connect to camera
capture = cv2.VideoCapture('markerTest.mov')
#capture = cv2.VideoCapture("markertest.mov")


while True:
	ret, img = capture.read()
	marker_corners, marker_id, rejected = detector.detectMarkers(img)
	print(marker_corners)
	cv2.aruco.drawDetectedMarkers(img, marker_corners)

	cv2.imshow("frame", img)

	# key 27 is esc
	if (cv2.waitKey(1) == 27):  # wait for 1 ms
		break

# allow other processes to use the camera after release
capture.release() 
cv2.destroyAllWindows