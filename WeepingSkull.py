'''
Author: Adrian Rosebrock
Projects Combined:
	- Simple-Object-Tracking : https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/?__s=zsmw7ek8fkeso4iokqco
	- Real-Time-Object-Detection : https://www.pyimagesearch.com/2017/10/16/raspberry-pi-deep-learning-object-detection-with-opencv/
Combiner and Modifier: Zayin Brunson
Purpose: Think of weeping angels from video games, and this is the same but with a skull head. It follows a person, and lights up its eyes, whenever it does not recognize a face.
'''
# USAGE
# py WeepingSkull.py --ip ###.###.###.### --source edison

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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True,
	help="Source of video stream (webcam/host)")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
	help="# of skip frames between detections")
ap.add_argument("--ip", required=True,
	help="The IP Address")
args = vars(ap.parse_args())

ipaddress = args["ip"]
url = 'http://'+ipaddress+'/html/cam_pic_new.php?'

ct = CentroidTracker() #Makes the magic happen
(H, W) = (None, None) #Sets the size of the image

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
#IGNORE all classes except for person
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

#Connect - Used to start the ssh session into the Raspberry Pi
#Param:
	#ip - the given ip as provided by args
#Return: variable that stores the ssh session
def Connect(ip, username='pi', pw='basicrandompassword'):
    ssh = SSHClient()
    ssh.set_missing_host_key_policy(AutoAddPolicy())
    ssh.connect(ip, username=username, password=pw)
    return ssh

#Connect - Sends a command via ssh to the Raspberry Pi Terminal/Bash
#Param:
	#ssh - ssh session variable
	#command - the command to be executed on the terminal
def SendCommand(ssh, command, pw='password'):
    stdin, stdout, stderr = ssh.exec_command( command )
    if "sudo" in command:
    	stdin.write(pw+'\n')
    stdin.flush()

myssh = Connect(ip=ipaddress) #SSH Session
SendCommand(myssh, command=('python ~/servomotor.py 90')) #resets head
SendCommand(myssh, command=('python ~/blink.py OFF')) #turns off lights
objects = ct #CentroidTracker()
prev = 0 #previous pixel location along the x axis
timerObjects = 50 #Loop ticks until lights will turn off
timeToReset = 500 #Loop ticks until motor resets to center
blink = "OFF" #Used so that the script doesn't spam the RPI3
OGStartX = 0 #Left side of detection
OGStartY = 0 #Upper side of detection
OGEndX = 0 #Right side of detection
OGEndY = 0 #Lower side of detection
wasFace = 0 #Used as a timer for the last time a face was seen
wasBody = 0 #Used as a timer for the last time a body was seen
prevRowCentroid = 0 #Used to store the previous iteration of centroids
prevColCentroid = 0 #Used to store the previous iteration of centroids
# loop over the frames from the video stream


min = 999999999999 #used to find the lowest possible objectId that exists

while True: # Forever loop

	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=800)
	if totalFrames % args["skip_frames"] == 0:
		# grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
			0.007843, (300, 300), 127.5)
		blob2 = cv2.dnn.blobFromImage(cv2.resize(frame, (600, 600)), 1.0,
			(600, 600), (104.0, 177.0, 123.0))

		# pass the blob through the network and obtain the detections and predictions
		net.setInput(blob) #For the body
		detections = net.forward()
		net2.setInput(blob2) #For the head
		detections2 = net2.forward()
		rects = []

		# loop over the detections
		if detections.shape[2] == 0:
			timerObjects-=1
			timeToReset-=1

		#No objects found, turn off lights
		if timerObjects < 0:
			timerObjects = 50
			if blink != "OFF":
				blink = "OFF"
				SendCommand(myssh, command=('python ~/blink.py OFF'))

		#too long to find anything, reset position
		if timeToReset < 0:
			timeToReset = 500
			SendCommand(myssh, command=('python ~/servomotor.py 90'))

		flag = True
		#Finds all possible bodys and adds them to be a object
		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			#Using the given confidence, check to see if it passes enought to be called a body
			if confidence > args["confidence"]:
				idx = int(detections[0, 0, i, 1])
				if CLASSES[idx] in IGNORE:
					continue
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				rects.append(box.astype("int"))
				(startX, startY, endX, endY) = box.astype("int")
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 255, 0), 2)

		# update our centroid tracker using the computed set of bounding box rectangles
		objects = ct.update(rects)
		min = 999999999999

		# loop over the tracked objects and gives them their UniqueID
		for (objectID, centroid) in objects.items():
			if objectID < min:
				min = objectID
				prevRowCentroid = centroid[0]
				prevColCentroid = centroid[1]
			# draw both the ID of the object and the centroid of the object on the output frame
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

		#if a minimum UniqueID exists and there is an existing object
		if min != 999999999999 and len(rects) > 0:
			wasBody = 10
			minBodyID = ct.getMinUpdate(rects,min,prevRowCentroid,prevColCentroid)
			#given the current body object of the smallest available ID, find a face with it
			if minBodyID[0] != "ERROR":
				OGStartX = minBodyID[1]
				OGEndX = minBodyID[3]
				OGStartY = minBodyID[2]
				OGEndY = minBodyID[4]

				#Tries to find all faces that are withing a body object found in detctions
				for i in range(0, detections2.shape[2]):
					confidence = detections2[0, 0, i, 2]
					# filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
					if confidence < args["confidence"]:
						continue

					# compute the (x, y)-coordinates of the bounding box for the object
					box = detections2[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					# draw the bounding box of the face
					text = "{:.2f}%".format(confidence * 100)
					y = startY - 10 if startY - 10 > 10 else startY + 10
					cv2.rectangle(frame, (startX, startY), (endX, endY),
						(0, 0, 255), 2)
					cv2.putText(frame, text, (startX, y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

					#if the face is outside of the body object turn off the lights, start the timer for wasFace, and set flag to False
					if  min == minBodyID[0] and startX > minBodyID[1] and endX < minBodyID[3] and startY+50 > minBodyID[2] and endY < minBodyID[4]:
						if blink != "OFF":
							blink = "OFF"
							SendCommand(myssh, command=('python ~/blink.py OFF'))
						wasFace = 10
						flag = False

		#No Body object exists currently
		else:
			flag = False
			if blink != "OFF" and wasBody <= 0:
				blink = "OFF"
				SendCommand(myssh, command=('python ~/blink.py OFF'))
			elif wasBody > 0:
				wasBody -= 1

		#If there was a face, then send ssh to the RPI to move it
		if flag and wasFace <= 0:
			if blink != "ON":
				blink = "ON"
				SendCommand(myssh, command=('python ~/blink.py ON'))
			xDist = 90 + ((400 - (OGStartX+OGEndX)/2.0)*30.5/400)
			if abs(xDist - prev) > 3:
				prev = xDist
				SendCommand(myssh, command=('python ~/servomotor.py '+str(xDist)))

		#If there was no face, count down timer
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
