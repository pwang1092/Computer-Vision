import cv2
import numpy as np

INPUT_WIDTH = 640
INPUT_HEIGHT = 640

CONFIDENCE_THRESHOLD = 0.45
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
BLUE = (255,0,0)
BLACK = (0,0,0)
THICKNESS = 1
YELLOW =(0,255,255)
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7


def draw_label(input_image, label, left, top):
	text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
	dim, baseline = text_size[0], text_size[1]
	cv2.rectangle(input_image, (left, top), (left+dim[0], top+dim[1]+baseline), BLACK, cv2.FILLED)
	cv2.putText(input_image, label, (left, top+dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)


def pre_process(input_image, net):
	blob = cv2.dnn.blobFromImage( input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop = False)
	net.setInput( blob )
	output_layers = net.getUnconnectedOutLayersNames()
	output = net.forward( output_layers )

	return output


def post_process(input_image, outputs):
	class_ids = []
	confidences = []
	boxes = []

	rows = outputs[0].shape[1]
	image_height, image_width = input_image.shape[:2]
	x_factor = image_width / INPUT_WIDTH
	y_factor = image_height / INPUT_HEIGHT

	for r in range(rows):
		row = outputs[0][0][r]
		confidence = row[4]

		#discard bad detections
		if confidence > CONFIDENCE_THRESHOLD:
			classes_scores = row[5:]
			class_id = np.argmax(classes_scores)
			if(classes_scores[class_id] > SCORE_THRESHOLD ):
				confidences.append(confidence)
				class_ids.append(class_id)
				cx, cy, w, h = row[0], row[1], row[2], row[3]
				left = int((cx-w/2) * x_factor)
				top = int((cy-h/2) * y_factor)
				width = int(w * x_factor)
				height = int(h * y_factor)
				box = np.array([left, top, width, height])
				boxes.append( box )
				print (box)


	#remove close similar detections
	#keep highest score
	indices = cv2.dnn.NMSBoxes( boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD )
	for i in indices:
		box = boxes[i]
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		cv2.rectangle(input_image, (left,top), (left+width, top+height), BLUE, 3)
		label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
		draw_label(input_image, label, left, top)
	return input_image

# load class names
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
	classes = f.read().rstrip('\n').split('\n')

# give weights to the model
modelWeights = "models/yolov5s.onnx"
net = cv2.dnn.readNet(modelWeights)

capture = cv2.VideoCapture(0)

while True:
	ret, img = capture.read()
	height,width,channels = img.shape
	#object detection
	detections = pre_process(img, net)
	img = post_process(img.copy(), detections)

	cv2.imshow("image", img)
	if (cv2.waitKey(1) == 27):
		break

capture.release()
cv2.destroyAllWindows() 
