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



#url = host + 'video'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
'''ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-p2", "--prototxt2", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m2", "--model2", required=True,
	help="path to Caffe pre-trained model")'''
ap.add_argument("--source", required=True,
	help="Source of video stream (webcam/host)")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
	help="# of skip frames between detections")
ap.add_argument("--ip", required=True,
	help="The IP Address")
args = vars(ap.parse_args())
#print(args)

ipaddress = args["ip"]
url = 'http://'+ipaddress+'/html/cam_pic_new.php?'

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
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
net2 = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")

if args["source"] == "webcam":
	vs = VideoStream(src=0).start()
elif args["source"] == "edison":
	vs = VideoStream(src=url).start()

time.sleep(2.0)
totalFrames = 0
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
SendCommand(myssh, command=('python ~/servomotor.py 90'))
SendCommand(myssh, command=('python ~/blink.py OFF'))
degree = 90
isMoving = False
prev = 0
testObjects = 0
timeToReset = 500
blink = "OFF"
OGStartX = 0
OGStartY = 0
OGEndX = 0
OGEndY = 0
wasFace = 0
wasBody = 0
prevID = 999999999999
wasPrevID = 0
prevRowCentroid = 0
prevColCentroid = 0
# loop over the frames from the video stream

min = 999999999999

while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	if frame == "None":
		print("DIED")
	print("THING::"+str(frame))
	frame = imutils.resize(frame, width=800)
	if totalFrames % args["skip_frames"] == 0:
		#if W is None or H is None:
		#	(H, W) = frame.shape[:2]

		# grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
			0.007843, (300, 300), 127.5)
		blob2 = cv2.dnn.blobFromImage(cv2.resize(frame, (600, 600)), 1.0,
			(600, 600), (104.0, 177.0, 123.0))
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
		#test = np.arange(0, detections.shape[2])
		#print(detections.shape[2])print
		if detections.shape[2] == 0:
			testObjects+=1
			timeToReset-=1
		if testObjects > 50:
			testObjects = 0
			if blink != "OFF":
				print("1")
				blink = "OFF"
				SendCommand(myssh, command=('python ~/blink.py OFF'))
		if timeToReset < 0:
			timeToReset = 500
			SendCommand(myssh, command=('python ~/servomotor.py 90'))

		flag = True
		#print("_______________________________")
		for i in np.arange(0, detections.shape[2]):
			#print(detections.shape)
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]
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
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 255, 0), 2)

		# update our centroid tracker using the computed set of bounding
		# box rectangles




		objects = ct.update(rects)
		min = 999999999999
		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			#print(objectID)
			if objectID < min:
				min = objectID
				prevRowCentroid = centroid[0]
				prevColCentroid = centroid[1]
			# draw both the ID of the object and the centroid of the
			# object on the output frame
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)




		if min != 999999999999 and len(rects) > 0:
			wasBody = 10
			minBodyID = ct.getMinUpdate(rects,min,prevRowCentroid,prevColCentroid)
			if minBodyID[0] != "ERROR":
				prevID = minBodyID[0]
				OGStartX = minBodyID[1]
				OGEndX = minBodyID[3]
				OGStartY = minBodyID[2]
				OGEndY = minBodyID[4]
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
					if  min == minBodyID[0] and startX > minBodyID[1] and endX < minBodyID[3] and startY+50 > minBodyID[2] and endY < minBodyID[4]:
						if blink != "OFF":
							blink = "OFF"
							SendCommand(myssh, command=('python ~/blink.py OFF'))
						wasFace = 10
						flag = False
			'''elif prevID != minBodyID[0]:
				wasPrevID -= 1'''
		else:
			flag = False
			if blink != "OFF" and wasBody <= 0:
				blink = "OFF"
				SendCommand(myssh, command=('python ~/blink.py OFF'))
			elif wasBody > 0:
				wasBody -= 1



		if flag and wasFace <= 0:
			if blink != "ON":
				blink = "ON"
				SendCommand(myssh, command=('python ~/blink.py ON'))
			xDist = 90 + ((400 - (OGStartX+OGEndX)/2.0)*30.5/400)
			if abs(xDist - prev) > 3:
				prev = xDist
				SendCommand(myssh, command=('python ~/servomotor.py '+str(xDist)))
		elif wasFace > 0:
			wasFace -= 1



	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	totalFrames += 1
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
