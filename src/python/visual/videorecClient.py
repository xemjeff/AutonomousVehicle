#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import logging
# cv2 and helper:
import cv2
import Image
import os
from socket import *
#import cv2
import numpy
from getpass import getpass
from matplotlib import pyplot as plt
from helper.common import *
from helper.video import *
# add facerec to system path
import sys
sys.path.append("../..")
# facerec imports
from facerec.model import PredictableModel
from facerec.feature import Fisherfaces
from facerec.distance import EuclideanDistance
from facerec.classifier import NearestNeighbor
from facerec.validation import KFoldCrossValidation
from facerec.serialization import save_model, load_model
# for face detection (you can also use OpenCV2 directly):
from facedet.detector import CascadedDetector
# tesseract for ocr
try: import Image
except: from PIL import Image
import pytesseract
import itertools
import redis
from time import sleep
from datetime import datetime
from datetime import timedelta
from collections import deque
import threading
import imutils

# Connect to the rPi
try:
	ip = os.popen('AV').read().split('Address: ')[1].replace('\n','')
except:
	print "The AV is offline"
	sys.exit () 

memory = redis.StrictRedis(ip,port=6379,db=0)
memory.set('current_state','default')

#Load the Caffe model 
net = cv2.dnn.readNetFromCaffe('/root/AutonomousVehicle/src/python/visual/MobileNetSSD_deploy.prototxt', '/root/AutonomousVehicle/src/python/visual/MobileNetSSD_deploy.caffemodel')

sleep(5)

n = 25

# Labels of Network.
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'dining table', 12: 'dog', 13: 'horse',
    14: 'motor bike', 15: 'person', 16: 'potted plant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tv monitor' }

states = ['default','person_features','vegetables','vehicles','signs']
default = ['person','face','car','bottle','bus','iphone','smile']
person_features = ['eye','eyeglasses','face_1','hand','left_eye','lower_body','mouth','nose','profile_face','right_eye','shoulders','smile','upper_body']
vegetables = ['apple','banana']
vehicles = ['car','bus','two_wheeler','licence_plate']
signs = ['signs','yield_sign','stop_sign','speed_sign']
current_definites = []

# Ball color range defined
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen=30)

descriptors = [['face_3',5,1.1],['eye',5,1.1],['hand',5,1.25],['smile',5,1.35],['cat_face_1',5,1.5],['lower_body',5,1.2]]

filenames = glob.glob('/root/AutonomousVehicle/src/python/visual/haarcascade/*')
filelist = [z.split('.xml')[0].split('/')[-1] for z in filenames]

# Create a socket object
sock1 = socket() 
sock2 = socket()
print '==' * n
sock1.connect((ip, 5000))
sock2.connect((ip, 5001))

# Disparity settings
window_size = 5
min_disp = 32
num_disp = 112-min_disp
stereo = cv2.StereoSGBM(
    minDisparity = min_disp,
    numDisparities = num_disp,
    SADWindowSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 1,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    fullDP = False
)
 
# Morphology settings
kernel = np.ones((12,12),np.uint8)

def recvall(conn, count):
	buf = b''
    	while count:
        	newbuf = conn.recv(count)
        	if not newbuf:
        		return None
        	buf += newbuf
        	count -= len(newbuf)
        return buf


class ExtendedPredictableModel(PredictableModel):
    """ Subclasses the PredictableModel to store some more
        information, so we don't need to pass the dataset
        on each program call...
    """

    def __init__(self, feature, classifier, image_size, subject_names):
        PredictableModel.__init__(self, feature=feature, classifier=classifier)
        self.image_size = image_size
        self.subject_names = subject_names

def get_model(image_size, subject_names):
    """ This method returns the PredictableModel which is used to learn a model
        for possible further usage. If you want to define your own model, this
        is the method to return it from!
    """
    # Define the Fisherfaces Method as Feature Extraction method:
    feature = Fisherfaces()
    # Define a 1-NN classifier with Euclidean Distance:
    classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)
    # Return the model as the combination:
    return ExtendedPredictableModel(feature=feature, classifier=classifier, image_size=image_size, subject_names=subject_names)

def read_subject_names(path):
    """Reads the folders of a given directory, which are used to display some
        meaningful name instead of simply displaying a number.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).

    Returns:
        folder_names: The names of the folder, so you can display it in a prediction.
    """
    folder_names = []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            folder_names.append(subdirname)
    return folder_names

def read_images(path, image_size=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes 

    Returns:
        A list [X, y, folder_names]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
            folder_names: The names of the folder, so you can display it in a prediction.
    """
    c = 0
    X = []
    y = []
    folder_names = []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            folder_names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    # resize to given size (if given)
                    if (image_size is not None):
                        im = cv2.resize(im, image_size)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c+1
    return [X,y,folder_names]


class videorecClient():

    def __init__(self):
        self.model = load_model('/root/AutonomousVehicle/src/python/visual/model.pkl')
	cascade_filename='/root/AutonomousVehicle/src/python/visual/haarcascade/face.xml'
	camera_id=0
	# Recursive call to load all cascades in memory
	for x in xrange(0,len(filelist)):
		if not filelist[x] in default:
			continue
		desc = [5,1.1]
		for y in xrange(0,len(descriptors)):
			if descriptors[y][0] == filelist[x]:
				desc[0] = descriptors[y][1]
				desc[1] = descriptors[y][2]
				break
		else:
			descriptors.append([filelist[x],5,1.1])
		exec('self.%s = CascadedDetector(cascade_fn="%s", minNeighbors=%s, scaleFactor=%s)' % (filelist[x],filenames[x],desc[0],desc[1]))
	threading.Thread(target=self.run).start()

    def run(self):
	avg = None

        while True:
	    # Negotiation for the sockets
            length1 = recvall(sock1, 16)
            if length1 == None:
                break
            length2 = recvall(sock2, 16)
            if length2 == None:
                break

	    # If the vision detection is activated and motion isn't being ignored
	    if True: #(memory.get('visionDetect') == 'True') and (memory.get('ignoreLastMotion') == 'False'):
		    buf1 = recvall(sock1, int(length1))
		    buf2 = recvall(sock2, int(length2))
		    data1 = numpy.fromstring(buf1, dtype='uint8')
		    data2 = numpy.fromstring(buf2, dtype='uint8')

		    frame1 = cv2.imdecode(data1, 1)
		    frame2 = cv2.imdecode(data2, 1)
		    
		    # Special analysis from frame1
		    img = cv2.resize(frame1, (frame1.shape[1], frame1.shape[0]), interpolation = cv2.INTER_CUBIC)
		    frame_resized = cv2.resize(frame1,(300,300))
		    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)

		    # Preprocessing for ball detection
		    blurred = cv2.GaussianBlur(frame1, (11, 11), 0)
		    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
		    mask = cv2.inRange(hsv, greenLower, greenUpper)
		    mask = cv2.erode(mask, None, iterations=2)
		    mask = cv2.dilate(mask, None, iterations=2)

		    #Set to network the input blob 
		    net.setInput(blob)
		    #Prediction of network
		    detections = net.forward()

		    #Size of frame resize (300x300)
		    cols = frame_resized.shape[1] 
		    rows = frame_resized.shape[0]

		    # Get the disparity from both cams
		    stereo = cv2.StereoBM_create(numDisparities=16,blockSize=15)
		    #stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET,ndisparities=16, SADWindowSize=15)
		    #stereo = cv2.createStereoBM(numDisparities=16, blockSize=15)
		    frame1_new = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
		    frame2_new = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
		    disparity = stereo.compute(frame1_new,frame2_new)

		    imgout = img.copy()

		    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		    gray = cv2.GaussianBlur(gray, (21,21), 0)

		    if avg is None:
		    	#print "[INFO] starting background model..."
			avg = gray.copy().astype("float")

		    cv2.accumulateWeighted(gray, avg, 0.5)
		    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

		    thresh = cv2.threshold(frameDelta, 5, 255,cv2.THRESH_BINARY)[1]
	     	    thresh = cv2.dilate(thresh, None, iterations=2)
		    _,cnts,ret, = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		    # Motion detect
		    current_motion = []
		    for c in cnts:
			# If the contour is too small, ignore it
			if cv2.contourArea(c) < 5000:
				continue
			(x, y, w, h) = cv2.boundingRect(c)
			c_x = ((x+(x+w))/2)*2
			c_y = ((y+(y+h))/2)*2
			current_motion.extend(['motion',c_x,c_y,100])
		        memory.set('last_motion_time',datetime.now())
			memory.set('motion_ended','False')
		    else:
			memory.set('motion_ended','True')

		    # Ball Detection
		    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		    center = None
		    ball = []
		    if len(cnts) > 0:
			c = max(cnts, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
			# Only proceed if the radius meets a minimum size (distance from AV)
			if radius > 10:
				ball = ['ball',int(x),int(y),100]

	 	    # Haarcascade analysis
		    # Recursive call to look for matching, selected cascades
		    current_frame_haar = []
		    current_state = memory.get('current_state')
		    filelist = []
		    exec('filelist = %s' % current_state)
		    for x in xrange(0, len(filelist)):
			    for i,r in enumerate(eval('self.%s' % filelist[x]).detect(img)):
				if memory.get('videoCascadesCalibrate') != 'True':
					# Initialize image, coords & prediction
					filename = filelist[x]
					x0,y0,x1,y1 = r
					exec('%s = img[y0:y1, x0:x1]' % filename)
					exec('%s = cv2.cvtColor(%s,cv2.COLOR_BGR2GRAY)' % (filename,filename))
					#exec('%s = cv2.resize(%s, self.model.image_size, interpolation = cv2.INTER_CUBIC)' % (filename,filename))
					# Get a prediction from the model:
					#exec('prediction = self.model.predict(%s)[0]' % filename)
					current_frame_haar.extend([[filename,((x0+x1)/2),((y0+y1)/2),100]])
				else:
					desc_num = (["'"+filelist[x]+"'" in str(s) for s in descriptors].index(True))
				
					if float(descriptors[desc_num][2]) > 1.01:
						#print "Filelist: " + str(filelist[x])
						#print "From "+str(descriptors[desc_num][2])
						descriptors[desc_num][2] = (float(descriptors[desc_num][2])-0.01)
						try:
							exec('self.%s = CascadedDetector(cascade_fn="%s", minNeighbors=%s, scaleFactor=%s)' % (filelist[x],filenames[x],descriptors[desc_num][1],descriptors[desc_num][2]))
							#print "To "+str(descriptors[desc_num][2])
						except:
							pass
					else:
						pass
					break
			    else:
				pass

		    current_frame_tensorflow = []
	 	    # Tensorflow analysis
		    for i in range(detections.shape[2]):
			confidence = detections[0, 0, i, 2] #Confidence of prediction 
			if confidence > 0.2: # Filter prediction 
			    class_id = int(detections[0, 0, i, 1]) # Class label
			    # Object location 
			    _xLeftBottom = int(detections[0, 0, i, 3] * cols) 
			    _yLeftBottom = int(detections[0, 0, i, 4] * rows)
			    _xRightTop   = int(detections[0, 0, i, 5] * cols)
			    _yRightTop   = int(detections[0, 0, i, 6] * rows)
			    # Factor for scale to original size of frame
			    heightFactor = frame1.shape[0]/300.0  
			    widthFactor = frame1.shape[1]/300.0 
			    # Scale object detection to frame
			    xLeftBottom = int(widthFactor * _xLeftBottom) 
			    yLeftBottom = int(heightFactor * _yLeftBottom)
			    xRightTop   = int(widthFactor * _xRightTop)
			    yRightTop   = int(heightFactor * _yRightTop)
			    # Label and confidence of prediction in frame resized
			    if class_id in classNames:
				label = classNames[class_id] + ": " + str(confidence)
				labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

				yLeftBottom = max(yLeftBottom, labelSize[1])
				current_frame_tensorflow.extend([[label,((_xLeftBottom+_xRightTop)/2),((_yLeftBottom+_yRightTop)/2),confidence]])

	    definites = []
	    # Combines haar & tensorflow objects, then finds repeated ones
	    combined = current_frame_haar.extend(current_frame_tensorflow)
	    if combined:
		    repeated = [z for z in combined if combined.count(z) > 1]
		    combined.sort()
		    combined = list(combined for combined,_ in itertools.groupby(combined))
		    repeated.sort()
		    repeated = list(repeated for repeated,_ in itertools.groupby(repeated))
		    # Check for high confidence
		    high_confidence = [z for z in combined if float(z[3]) > 99.1]
		    if high_confidence: repeated.extend(high_confidence)
		    # Gets speed limit from road signs
		    if [z for z in combined if 'sign' in z[0]]:
			text = pytesseract.image_to_string(imgout)
			if text: memory.set('speed_sign_text',text)
		    # Repeated & high confidence detected objects are treated as foolproof
		    if repeated:
			for x in xrange(0,len(repeated)):
				# Add them to definite objects currently detected, if not already there
				if not [z for z in current_definites if repeated[x][0] in current_definites]:
					definites.extend(repeated[x])
	    # Checks if there is a ball
	    if ball: definites.extend(ball)
	    if current_motion: definites.extend(current_motion)
	    # Removes objects not currently detected from definite objects list
	    current_definites = definites

	    # Adds the distance from each object
	    #current_definites = self.add_depth_variable(current_definites)

	    memory.set('objects_detected',str(current_definites))

	    sock1.send('OK')
	    sock2.send('OK')
