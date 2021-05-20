import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
from pyimagesearch.centroidtracker import CentroidTracker
from collections import deque

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

#dictionary to store the positions of each object
dic = {}
#dict to store whether an object is reversing or not
is_rev = {}
#dict to know if an object is slowing down/ halted
halted = {}
#dict to store the drn of motion of objects
d1 = {} #
d2 = {} #if an object halts, what was its direction so far
d3 = {} #key = object id, value = a number corresponding to direction to check for




parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--device', default='cpu', help="Device to perform inference on 'cpu' or 'gpu'.")
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

#set up the model
classesFile = 'coco.names'
classes =  None
with open(classesFile,'rt') as f:
	classes = f.read().split('\n') #f.read() contains the entire contents of the file

#load the model
modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv.dnn.readNetFromDarknet(modelConfiguration,modelWeights)

if(args.device == 'cpu'):
	net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
	net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
	print('Using CPU device.')
elif(args.device == 'gpu'):
	net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
	print('Using GPU device.')



	
# Get the names of the output layers
def getOutputsNames(net):
	# Get the names of all the layers in the network
	layersNames = net.getLayerNames()
	# Get the names of the output layers, i.e. the layers with unconnected outputs
	return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]    


def postprocess(frame,outs):
	global dic
	global is_rev
	frameHeight = frame.shape[0]
	frameWidth = frame.shape[1]
	
	# Scan through all the bounding boxes output from the network and keep only the
	# ones with high confidence scores. Assign the box's class label as the class with the highest score.
	classIds = []
	confidences = []
	boxes = []
	boxes_2 = []
	dic1 = {}
	dic2 = {}
	i = 0
	for out in outs:
		for detection in out:
			scores = detection[5:]
			classId = np.argmax(scores)
			confidence = scores[classId]
			if confidence > confThreshold:
				center_x = int(detection[0] * frameWidth)
				center_y = int(detection[1] * frameHeight)
				width = int(detection[2] * frameWidth)
				height = int(detection[3] * frameHeight)
				left = int(center_x - width / 2)
				top = int(center_y - height / 2)
				classIds.append(classId)
				confidences.append(float(confidence))
				box = [left, top, left+width, top+height]
				dic1[i]=box
				dic2[i]=[left,top,width,height]
				i+=1
				boxes.append([left,top,width,height])
				# cv.rectangle(frame, (left, top), (left+width, top+height),(0, 255, 0), 2)

	# Perform non maximum suppression to eliminate redundant overlapping boxes with
	# lower confidences.
	indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

	
	for i in indices:
		i = i[0]
		bx = boxes[i]
		for k in dic2.keys():
			if dic2[k] ==  bx:
					boxes_2.append(dic1[k])
	boxes.clear()
	boxes = boxes_2

	objects = ct.update(boxes)
	# loop over the tracked objects
	i = 0
	for (objectID, centroid) in objects.items():
		#draw the bounding box
		if len(boxes) > i:
			box = boxes[i]
			i+=1
			cv.rectangle(frame,(box[0],box[1]),(box[2],box[3]),(0, 255, 0), 2)

		if objectID not in dic:
			dic[objectID] = deque([])
			if centroid[1] < frameHeight/4:
				d3[objectID] = 0
			elif centroid[1]> frameHeight/3 and centroid[0] < frameWidth/4:
				d3[objectID] = 1
			else:
				d3[objectID] = 2

		dic[objectID].appendleft(centroid)
		print(centroid)

		nframes = 25
		variance_limt1 = 100
		# variance_limt2 = 300

		if len(dic[objectID]) == nframes : #time to analyze the motion
			x = []
			y = []
			for i in range(nframes):
				x.append(dic[objectID][i][0])
				y.append(dic[objectID][i][1])
			if objectID == 0:
				print('xvar {} yvar {}'.format(np.var(x),np.var(y)))

			if objectID not in d1:
				d1[objectID] = [0,0,0,0]

			c1 = dic[objectID][-1]
			cn = dic[objectID][0]
			# drn = ''
			if d3[objectID] == 0:
				if cn[1] > c1[1]:
					# drn = 'fwd'
					d1[objectID][0]+=1
				else :
					# drn = 'bcwd'
					d1[objectID][0]-=1
			elif d3[objectID] == 1:
				if cn[0] > c1[0]:
					#drn = 'moving right'
					d1[objectID][1] +=1
				else :
					d1[objectID][1] -=1
			else:
				if cn[0] < c1[0]:
						#drn = 'moving left'
					d1[objectID][2] +=1
				else :
					d1[objectID][2] -=1

			if objectID not in halted :
				halted[objectID]= 0
			
			if halted[objectID]: #the object had halted for 1 second or so
				#observe the direction for 3 sets of 25 frames
				#if the final direction after 3*25 = 75 frames was opposite to that before halting, then this could be reversing 
				if d1[objectID][3] == 3 : 
					if d2[objectID]*d1[objectID][d3[objectID]]<=0 and d2[objectID]>0: 
						is_rev[objectID]=1
					d1[objectID][3]= 0
					halted[objectID] = 0
				else :
					d1[objectID][3]+=1
				
				
			if objectID in is_rev:
				p1 = is_rev[objectID]
			else :
				p1 = 0
			if objectID == 0:
				print('rev {} drn {}'.format(p1,d1[objectID]))
			if np.var(x) + np.var(y) < variance_limt1 and halted[objectID]== 0 :
				# print(1)
				halted[objectID]=1
				d2[objectID] = d1[objectID][d3[objectID]]
				d1[objectID] = [0,0,0,0] 
					
					
			for i in range(13):
				dic[objectID].pop()
			# dic[objectID].clear()

				
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		flag = 0
		color = (0,255,0)
		if objectID in is_rev:
			flag = 1
			color = (0,0,255)
		text = "ID {} Rev {}".format(objectID,flag)
		cv.putText(frame, text, (centroid[0]-40, centroid[1] + 40),cv.FONT_HERSHEY_SIMPLEX, 1.5, color, 6)
		cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

		
						




# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"
if (args.image):
	# Open the image file
	if not os.path.isfile(args.image):
		print("Input image file ", args.image, " doesn't exist")
		sys.exit(1)
	cap = cv.VideoCapture(args.image)
	outputFile = args.image[:-4]+'_yolo_out_py.jpg'
elif (args.video):
	# Open the video file
	if not os.path.isfile(args.video):
		print("Input video file ", args.video, " doesn't exist")
		sys.exit(1)
	cap = cv.VideoCapture(args.video)
	outputFile = args.video[:-4]+'_yolo_out_py.avi'
else:
	# Webcam input
	cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
	vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:
	
	# get frame from the video
	hasFrame, frame = cap.read()
	
	# Stop the program if reached end of video
	if not hasFrame:
		print("Done processing !!!")
		print("Output file is stored as ", outputFile)
		cv.waitKey(3000)
		# Release device
		cap.release()
		break

	# Create a 4D blob from a frame.
	blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

	# Sets the input to the network
	net.setInput(blob)

	# Runs the forward pass to get output of the output layers
	outs = net.forward(getOutputsNames(net))

	# Remove the bounding boxes with low confidence
	postprocess(frame, outs)

	# Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
	t, _ = net.getPerfProfile()
	label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
	cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

	# Write the frame with the detection boxes
	if (args.image):
		cv.imwrite(outputFile, frame.astype(np.uint8))
	else:
		vid_writer.write(frame.astype(np.uint8))

	cv.imshow(winName, frame)







