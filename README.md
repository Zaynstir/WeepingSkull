# WeepingSkull
Combiner and Modifier: Zayin Brunson

Adrian Rosebrock is the creator of PyImageSearch and main projects that I combined in order to complete this project.
Projects Combined:
 - Simple-Object-Tracking : https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/?__s=zsmw7ek8fkeso4iokqco
 - Real-Time-Object-Detection : https://www.pyimagesearch.com/2017/10/16/raspberry-pi-deep-learning-object-detection-with-opencv/

Purpose: It was a fun side project aside from school work. it's only purpose was to be creepy to those who walked by. It would use PyImageSearch motion tracking to analyze video feed picked up by a raspberry pi camera module and then rotate a server motor with a halloween prop skull connected to it to simulate the skull watching people as they walked by. It would stop following a person if the person looked at it. The skull acts as a weeping angel from a variety of Sci-Fi movie and video games. In short, the skull will follow a person as long as it cannot recognize a face from that person.

This used many different parts
- Raspberry Pi 3
- RPI Camera module
- Servo Motor
- A separate host computer

I tried to have everything computed on the RPI3, but it proved too intense for the RPI, so I had the rpi livestream the camera feed to a local IP address. I ran the main python script on my computer which read the livestream and analyzed the data, and sent the computed results back to the rpi through ssh commands to rotate the servo motor or turn the lights that were embedded into the eyes on/off.
