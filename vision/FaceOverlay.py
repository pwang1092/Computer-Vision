import cv2
import numpy as np

#https://s.evodyne.co/arl/overlay.jpg
# Load the dictionary that was used to generate the markers
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
face_detector = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml") 

#bbox bounding box

def augment(bbox, id, img, img_aug, drawId = True):
	tl = bbox[0][0][0], bbox[0][0][1]
	tr = bbox[0][1][0], bbox[0][1][1]
	br = bbox[0][2][0], bbox[0][2][1]
	bl = bbox[0][3][0], bbox[0][3][1]
	h, w, c = img_aug.shape
	pts1 = np.array([tl,tr,br,bl]) #convert to np array
	pts2 = np.float32([[0,0], [w,0], [w,h], [0,h]])
	matrix, _ = cv2.findHomography(pts2, pts1)
	# warp image like warp of tile and fill tile
	imgout = cv2.warpPerspective(img_aug, matrix, (img.shape[1], img.shape[0])) 
	cv2.fillConvexPoly(img, pts1.astype(int), (0,0,0))
	imgout = img + imgout
	return imgout


# connect to camera
capture = cv2.VideoCapture(0)
#capture = cv2.VideoCapture("markertest.mov")
img_aug = cv2.imread("haley.jpg")

while True:
	ret, img = capture.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # detector needs greyscale
	faces = face_detector.detectMultiScale(gray) 

	# detect the marker
	marker_corners, marker_id, rejected = detector.detectMarkers(img)
	print(marker_corners)
	cv2.aruco.drawDetectedMarkers(img, marker_corners)

	for j in range(len(marker_corners)):
		if (j < len(faces)): 
			face = img[faces[j][1]:(faces[j][1]+faces[j][3]), faces[j][0]:(faces[j][0]+faces[j][2])]
		img = augment(marker_corners[j], marker_id[j], img, face)

	cv2.imshow("frame", img)

	# key 27 is esc
	if (cv2.waitKey(1) == 27):  # wait for 1 ms
		break

# allow other processes to use the camera after release
capture.release() 
cv2.destroyAllWindows



