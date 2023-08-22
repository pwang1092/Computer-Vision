import cv2
import numpy as np

#https://s.evodyne.co/arl/overlay.jpg
# Load the dictionary that was used to generate the markers
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
marker_size = 169

#information: https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec
CAM_CALIB = np.array(
	[[998, 0.0, 640],
	[0.0, 998, 360],
	[0.0, 0.0, 1.0]])

DIST_COEFFS = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

def localize(bbox, id, img):
	global marker_size
	# rotation vector and position vector? 
	rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(bbox, marker_size, CAM_CALIB, DIST_COEFFS)
	distance = np.sqrt(tvec[0][0][0]**2 + tvec[0][0][1]**2 + tvec[0][0][2]**2) #distance in 3d pythagorean
	print("Distance: ", distance)
	#documentation: https://shimat.github.io/opencvsharp_docs/html/3baec2d3-bb00-26ff-75a9-be6857745d08.htm
	cv2.drawFrameAxes(img, CAM_CALIB, DIST_COEFFS, rvec, tvec, 40)

	#defining an arbitrary point
	object_points = np.float32([[0, 0, 60]])
	image_points, _ =cv2.projectPoints(object_points, rvec, tvec, CAM_CALIB, DIST_COEFFS)
	point = (int(image_points[0][0][0])), int(image_points[0][0][1])
	cv2.circle(img, point, 5, (225, 225, 0))

	#pyramid lines
	front_bleft = [-30, -30, 0]
	front_tleft = [-30, -30, 60]
	front_bright = [-30, 30, 0]
	front_tright = [-30, 30, 60]
	back_bright = [30, 30, 0]
	back_tright = [30, 30, 60]
	back_bleft = [30, -30, 0]
	back_tleft = [30, -30, 60]
	top = [0, 0, 60]
	square_base = [front_bleft, front_bright, back_bright, back_bleft, front_bleft]
	front_side = [front_tleft, front_tright, front_bright, front_bleft, front_tleft]
	back_side = [back_tleft, back_tright, back_bright, back_bleft, back_tleft]
	top_side = [front_tleft, front_tright, back_tright, back_tleft, front_tleft]
	object_line_segments = [square_base, front_side, back_side, top_side]
	for object_points in object_line_segments:
		# documentation: https://amroamroamro.github.io/mexopencv/matlab/cv.projectPoints.html
		image_points, _ = cv2.projectPoints(np.float32(object_points), rvec, tvec, CAM_CALIB, DIST_COEFFS)
		print(image_points)
		draw_points = []
		for point in image_points:
			point2d = [int(point[0][0]), int(point[0][1])]
			draw_points.append(point2d)
			print(point2d)
		draw_points = np.array(draw_points)
		draw_points = draw_points.reshape((-1, 1, 2))
		cv2.polylines(img, [draw_points], False, (0, 0, 225), 2)

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



#measurement in nm

# connect to camera
capture = cv2.VideoCapture(0)
#capture = cv2.VideoCapture("markertest.mov")
#img_aug = cv2.imread("haley.jpg")

while True:
	ret, img = capture.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # detector needs greyscale
	faces = face_detector.detectMultiScale(gray) 

	# detect the marker
	marker_corners, marker_id, rejected = detector.detectMarkers(img)
	print(marker_corners)
	cv2.aruco.drawDetectedMarkers(img, marker_corners)

	for i in range(len(marker_corners)):
		localize(marker_corners[i], marker_id[i], img) 

	cv2.imshow("frame", img)

	# key 27 is esc
	if (cv2.waitKey(1) == 27):  # wait for 1 ms
		break

# allow other processes to use the camera after release
capture.release() 
cv2.destroyAllWindows



