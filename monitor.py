from libs import detection
import numpy as np
import argparse
import imutils
import cv2
import os
from scipy.spatial import distance
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
    	# return True
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False


USE_GPU = False

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="", help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="", help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1, help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = 'models/coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = 'models/yolov4-tiny.weights'
configPath = 'models/yolov4-tiny.cfg'

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if USE_GPU:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None
fps = int(vs.get(cv2.CAP_PROP_FPS))


def mouse_drawing(event, x, y, flags, params):
	if event == cv2.EVENT_LBUTTONDOWN:
		# print("Left click")
		if len(points) < 5:
			points.append((x, y))

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_drawing)


# loop over the frames from the video stream
iframe = 0
skipframe = 30
points = []
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# resize the frame and then detect people (and only people) in it
	# frame = imutils.resize(frame, width=700)
	(H, W) = frame.shape[:2]

	for point in points:
		cv2.circle(frame, point, 5, (0, 0, 255), -1)
		
	if iframe % skipframe == 0:
		objects = ['person']
		results = detection.detect_object(frame, net, ln, Idxs=[LABELS.index(i) for i in objects if LABELS.index(i) is not None])

	if len(points) == 5:

		# loop over the results
		orang = 0
		for (i, (classID, prob, bbox, centroid)) in enumerate(results):
			# extract the bounding box and centroid coordinates, then
			# initialize the color of the annotation
			(startX, startY, endX, endY) = bbox
			(cX, cY) = centroid
			
			dY = (endY - startY) // 3
			tmp = [centroid[0], centroid[1] + dY]
			if tmp[1] > H:
				tmp[1] = H


			polygon = Polygon(points)
			point = Point(tmp[0], tmp[1])
			if polygon.contains(point):
				cv2.circle(frame, (tmp[0], tmp[1]), 4, (0, 255, 255), -1)
				orang += 1
				# get the width and height of the text box
				text = 'ORANG'
				font_scale = 1.25
				font = cv2.FONT_HERSHEY_PLAIN
				color = (0,212,255)
				rectangle_bgr = color
				(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
				# set the text start position
				# y = startY - 10 if startY - 10 > 10 else startY + 10
				text_offset_x, text_offset_y = startX, startY-5
				# make the coords of the box with a small padding of two pixels
				box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 10, text_offset_y - text_height - 10))
				overlay = frame.copy()
				cv2.rectangle(overlay, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
				cv2.rectangle(overlay, (startX, startY), (endX, endY), rectangle_bgr, cv2.FILLED)
				cv2.putText(overlay, text, (text_offset_x+5, text_offset_y-5), font, fontScale=font_scale, color=(0,0,0), thickness=2)
				# apply the overlay
				alpha=0.6
				cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

				rectangle_bgr = (0,0,255)
				# make the coords of the box with a small padding of two pixels
				overlay = frame.copy()
				cv2.rectangle(overlay, (300,5), (580,35), rectangle_bgr, -1)
				# opacity
				alpha = 0.75
				# loop over the info tuples and draw them on our frame
				font_scale = .8
				font = cv2.FONT_ITALIC
				text = f'{orang} ORANG TERDETEKSI'
				cv2.putText(overlay, text, (300, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
				cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
			else:
				cv2.circle(frame, (tmp[0], tmp[1]), 4, (255, 255, 0), -1)
		
		# draw lines
		lcolor = (0, 0, 255)
		if iframe%5 == 0:
			lcolor = (150, 150, 250)
		cv2.line(frame, points[0], points[1], lcolor, 2)
		cv2.line(frame, points[1], points[2], lcolor, 2)
		cv2.line(frame, points[2], points[3], lcolor, 2)
		cv2.line(frame, points[3], points[4], lcolor, 2)
		cv2.line(frame, points[4], points[0], lcolor, 2)

	# check to see if the output frame should be displayed to our screen
	if args["display"] > 0:
		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# if an output video file path has been supplied and the video writer has not been initialized, do so now
	if args["output"] != "" and writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)

	# if the video writer is not None, write the frame to the output video file
	if writer is not None:
		writer.write(frame)

	iframe += 1

	if iframe > 3000:
		iframe = 0

