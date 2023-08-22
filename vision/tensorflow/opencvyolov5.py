import cv2
import numpy as np

with open('object_detection_classes_coco.txt', 'r') as f:
	class_names = f.read().split('\n')

model = cv2.dnn.readNet( model='frozen_inference_graph.pb', config = 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', framework = "TensorFlow")

capture = cv2.VideoCapture(0)
while True:
	ret, img = capture.read()
	height,width,channels = img.shape
	blob = cv2.dnn.blobFromImage( image=img, size=(100,100), mean=(104,117,123), swapRB=True)
	model.setInput( blob )
	output = model.forward() 

	for detection in output[0,0, :, :]:
		confidence = detection[2]
		if (confidence > 0.4):
			class_id = detection[1]
			class_name = class_names[int(class_id)-1]
			color = (0,0,255)
			box_x 		= detection[3] * width
			box_y 		= detection[4] * height
			box_width 	= detection[5] * width
			box_height	= detection[6] * height;
			cv2.rectangle( img, (int(box_x), int(box_y), int(box_width), int(box_height)), color, thickness = 2)
			cv2.putText( img, class_name, (int(box_x), int(box_y-5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

	cv2.imshow("image", img)
	if (cv2.waitKey(1) == 27):
		break

capture.release()
cv2.destroyAllWindows()
