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



ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True,
	help="Source of video stream (webcam/host)")
ap.add_argument("--ip", required=True,
	help="The IP Address")
args = vars(ap.parse_args())

ipaddress = args["ip"]
url = 'http://'+ipaddress+'/html/cam_pic_new.php?'

print("[INFO] starting video stream...")
'''if args["source"] == "webcam":
	vs = VideoStream(src=0).start()
elif args["source"] == "edison":
	vs = VideoStream(src=url).start()'''
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

myssh = Connect(ip=ipaddress)
prevX = 200
prev = 200
blink = "OFF"
SendCommand(myssh, command=('python ~/blink.py OFF'))
SendCommand(myssh, command=('python ~/servomotor.py 90'))


def on_click(event, x, y, flags, param):
    global prev, prevX, blink
    if event == cv2.EVENT_LBUTTONDOWN and x <= 400 and y <= 400 and y >50 and x != prevX:
        xDist = 90 + ((200 - x)*31.1/200)
        if xDist > 90:
            print("LEFT::"+str(xDist))
        else:
            print("RIGHT::"+str(xDist))
        if abs(xDist - prev) > 1:
            prev = xDist
            SendCommand(myssh, command=('python ~/servomotor.py '+str(xDist)))
    if event == cv2.EVENT_LBUTTONDOWN and x < 50 and y <= 50:
        if blink != "OFF":
            blink = "OFF"
            SendCommand(myssh, command=('python ~/blink.py OFF'))
            print("OFF")
        elif blink != "ON":
            blink = "ON"
            SendCommand(myssh, command=('python ~/blink.py ON'))
            print("ON")


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
