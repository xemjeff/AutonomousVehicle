import glob,logging,cv2,os,sys,itertools,redis,threading,imutils,numpy,sys
from time import sleep
from copy import copy

# Ball color range defined
orangeUpper = (30,255,255)
orangeLower = (5,131,172)
#orangeUpper = (64,255,255)
#orangeLower = (29,86,6)

cams = [z for z in glob.glob('/dev/*') if z.startswith('/dev/video')]

class onboard_video():
    ball = []
    cam_type='mono'
    upside_down = False
    count = 0

    def __init__(self,cam_type='mono'):
	self.cam_type = cam_type
	print "Models loaded: Starting streaming thread"
	threading.Thread(target=self.run).start()

    def run(self):
	cam1 = cv2.VideoCapture(int(cams[0][-1]))
        while True:
	    success, frame1 = cam1.read()
	    # Flips the img  
	    #if self.upside_down == 'False':
	    #frame1 = cv2.flip(img,1)
    	    # Preprocessing for ball detection
	    #orig = frame1
	    #frame1 = imutils.resize(frame1, width=600)
	    blurred = cv2.GaussianBlur(frame1, (11, 11), 0)
	    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	    mask = cv2.inRange(hsv, orangeLower, orangeUpper)
	    mask = cv2.erode(mask, None, iterations=2)
	    mask = cv2.dilate(mask, None, iterations=2)
	    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	    center = None
	    self.ball = []
	    # If there's at least one ball
	    if len(cnts) > 0:
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		# Only proceed if the radius meets a minimum size (distance from AV)
		#if radius > 5:
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		self.ball=[str(center[0]),str(center[1]),str(radius)]
	    self.count += 1
	    #cv2.imshow('test',orig)
	    print str(self.ball)
	    #if cv2.waitKey(1) == 27: break


