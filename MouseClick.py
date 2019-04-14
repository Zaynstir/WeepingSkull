'''
Author: Adrian Rosebrock
Projects Combined:
	- Simple-Object-Tracking : https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/?__s=zsmw7ek8fkeso4iokqco
	- Real-Time-Object-Detection : https://www.pyimagesearch.com/2017/10/16/raspberry-pi-deep-learning-object-detection-with-opencv/
Combiner and Modifier: Zayin Brunson
Purpose: Same functionality, but instead of image processing, you click on the pop-up display of the camera and it will look in that direction.
'''
# USAGE
# py MouseClick.py --ip ###.###.###.### --source edison

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
ap.add_argument("--ip", required=True,
	help="The IP Address")
args = vars(ap.parse_args())

ipaddress = args["ip"]
url = 'http://'+ipaddress+'/html/cam_pic_new.php?'

vs = VideoStream(src=url).start()
time.sleep(2.0)
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
prevX = 200 #previous place clicked | default center
prev = 200 #previous place clicked | default center
blink = "OFF"
SendCommand(myssh, command=('python ~/blink.py OFF')) #turns off lights
SendCommand(myssh, command=('python ~/servomotor.py 90')) #resets head

#OnClick - Sends a command via ssh to the Raspberry Pi Terminal/Bash
#Param:
	#event - eventListener on when someone clicked
	#x - x position of mouseclick
	#y - y position of mouseclick
	#flags - IDK
	#param - IDK
def on_click(event, x, y, flags, param):
    global prev, prevX, blink

	#On click the skull will move to look that direction
    if event == cv2.EVENT_LBUTTONDOWN and x <= 400 and y <= 400 and y >50 and x != prevX:
        xDist = 90 + ((200 - x)*31.1/200)
        if xDist > 90:
            print("LEFT::"+str(xDist))
        else:
            print("RIGHT::"+str(xDist))
        if abs(xDist - prev) > 1:
            prev = xDist
            SendCommand(myssh, command=('python ~/servomotor.py '+str(xDist)))
	#On click in the upper left hand corner, it switches the eyes on and off
    if event == cv2.EVENT_LBUTTONDOWN and x < 50 and y <= 50:
        if blink != "OFF":
            blink = "OFF"
            SendCommand(myssh, command=('python ~/blink.py OFF'))
            print("OFF")
        elif blink != "ON":
            blink = "ON"
            SendCommand(myssh, command=('python ~/blink.py ON'))
            print("ON")


#starts running everything
image = VideoStream(src=url).start()
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_click)

while True:

    frame = vs.read()
    #print(frame)
    frame = imutils.resize(frame, width=400)
    cv2.imshow("image", frame)
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
