# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
from imutils.video import FPS
from paramiko import SSHClient, AutoAddPolicy
import sys
import numpy as np
import argparse
import imutils
import time
import cv2
import math
from urllib.request import urlopen
ipaddress = "10.132.177.18"
host = 'http://'+ipaddress+':8081/'
url = host + 'video'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-p2", "--prototxt2", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m2", "--model2", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("--source", required=True,
	help="Source of video stream (webcam/host)")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
#print(args)



ct = CentroidTracker()
(H, W) = (None, None)
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
IGNORE = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
net2 = cv2.dnn.readNetFromCaffe(args["prototxt2"], args["model2"])

initial_milli_sec = int(round(time.time() * 1000))
# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")

if args["source"] == "webcam":
	vs = VideoStream(src=0).start()
elif args["source"] == "edison":
	vs = VideoStream(src=url).start()
time.sleep(2.0)
fps = FPS().start()

def Connect(ip, username='pi', pw='davidisg00d'):
    '''ssh into the pi'''
    #print('connecting to {}@{}...'.format(username, ip))
    ssh = SSHClient()
    ssh.set_missing_host_key_policy(AutoAddPolicy())
    ssh.connect(ip, username=username, password=pw)
    #print('connection status =', ssh.get_transport().is_active())
    return ssh

def SendCommand(ssh, command, pw='password'):
    '''send a terminal/bash command to the ssh'ed-into machine '''
    #print('sending a command... ', command)
    stdin, stdout, stderr = ssh.exec_command( command )
    if "sudo" in command:
    	stdin.write(pw+'\n')
    stdin.flush()
    #print('\nstout:',stdout.read())
    #print('\nsterr:',stderr.read())

myssh = Connect(ip=ipaddress)
objects = ct
# loop over the frames from the video stream

min = 999999999999

while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	#if W is None or H is None:
	#	(H, W) = frame.shape[:2]

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)
	blob2 = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
	#print(blob)
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	net2.setInput(blob2)
	detections2 = net2.forward()
	#print(detections2)
	rects = []
	#print(detections)
	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		#print(detections.shape)
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]
		flag = True
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			if CLASSES[idx] in IGNORE:
				continue
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			rects.append(box.astype("int"))
			(startX, startY, endX, endY) = box.astype("int")
			#print("TX:"+str(targetX)+"-TY:"+str(targetY))
			# draw the prediction on the frame
			#label = "{}: {:.2f}%".format(CLASSES[idx],
				#confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)
			OGStartX = startX
			OGStartY = startY
			OGEndX = endX
			OGEndY = endY
			#print(rects)
			#y = startY - 15 if startY - 15 > 15 else startY + 15
			#cv2.putText(frame, label, (startX, y),
				#cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			#FACE DETECTION

			if min != 999999999999:
				minBodyID = ct.getMinUpdate(rects,min)
				#if minBodyID[0] != "ERROR":
				print(minBodyID)
				if minBodyID[0] != "ERROR":
					for i in range(0, detections2.shape[2]):
						# extract the confidence (i.e., probability) associated with the
						# prediction
						confidence = detections2[0, 0, i, 2]

						# filter out weak detections by ensuring the `confidence` is
						# greater than the minimum confidence
						if confidence < args["confidence"]:
							continue

						# compute the (x, y)-coordinates of the bounding box for the
						# object
						box = detections2[0, 0, i, 3:7] * np.array([w, h, w, h])
						(startX, startY, endX, endY) = box.astype("int")

						# draw the bounding box of the face along with the associated
						# probability
						text = "{:.2f}%".format(confidence * 100)
						y = startY - 10 if startY - 10 > 10 else startY + 10
						cv2.rectangle(frame, (startX, startY), (endX, endY),
							(0, 0, 255), 2)
						cv2.putText(frame, text, (startX, y),
							cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
						#flag = True
						if  min == minBodyID[0] and startX > OGStartX and endX < OGEndX and startY+50 > OGStartY and endY < OGEndY:
						#	print("inside")
							print("false2")
							flag = False
				else:
					print("False1")
					'''flag = False
					for i in range(0, detections2.shape[2]):
						# extract the confidence (i.e., probability) associated with the
						# prediction
						confidence = detections2[0, 0, i, 2]

						# filter out weak detections by ensuring the `confidence` is
						# greater than the minimum confidence
						if confidence < args["confidence"]:
							continue

						# compute the (x, y)-coordinates of the bounding box for the
						# object
						box = detections2[0, 0, i, 3:7] * np.array([w, h, w, h])
						(startX, startY, endX, endY) = box.astype("int")

						# draw the bounding box of the face along with the associated
						# probability
						text = "{:.2f}%".format(confidence * 100)
						y = startY - 10 if startY - 10 > 10 else startY + 10
						cv2.rectangle(frame, (startX, startY), (endX, endY),
							(0, 0, 255), 2)
						cv2.putText(frame, text, (startX, y),
							cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)'''
		if flag and min != 999999999999:
			milli_sec = int(round(time.time() * 1000))
			if milli_sec - initial_milli_sec >= 100:
				#print("1000")
				initial_milli_sec = int(round(time.time() * 1000))
				xDist = 180 - ((OGEndX - OGStartX)/2 + OGStartX)*180/400
				#xDist = (str(OGStartX)+":"+str(OGEndX)+"->"+str((OGEndX - OGStartX)/2))
				#print(xDist)
				SendCommand(myssh, command=('python ~/servomotor.py '+str(xDist)))

	min = 999999999999
	# update our centroid tracker using the computed set of bounding
	# box rectangles
	objects = ct.update(rects)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		#print(objectID)
		if objectID < min:
			min = objectID
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		#print("CENT::"+str(centroid[0])+":"+str(centroid[1])+"-"+str(objectID))
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)



	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
