import glob,logging,cv2,os,sys,pytesseract,itertools,redis,threading,imutils,numpy,sys
from socket import *
from getpass import getpass
from matplotlib import pyplot as plt
from helper.common import *
from helper.video import *
#from lane_detection.line_fit_frames import laneDetect
# add facerec to system path
sys.path.append("/Users/jeff/Code/harrisonscode/AutonomousVehicle/src/python/visual/facerec/py/")
#from facerec.model import PredictableModel
#from facerec.feature import Fisherfaces
#from facerec.distance import EuclideanDistance
#from facerec.classifier import NearestNeighbor
#from facerec.validation import KFoldCrossValidation
#from facerec.serialization import save_model, load_model
#from facedet.detector import CascadedDetector
from time import sleep
from collections import deque
from stereovision.calibration import StereoCalibration
from copy import copy
from ParticleFilter import ParticleFilter

# Connect to the rPi
print "Videorec is finding the IP - may take 10 seconds"
os = os.popen('uname -a').read()
if 'Linux' in os: gw = os.popen('ip route | grep default').read().split('via ')[1].split(' dev')[0]
else: gw = os.popen('netstat -nr | grep "^default"').read()
ip = os.popen('nmap -p 5000 '+gw+'/24').read().split('raspberrypi.home (')[1].split(')')[0]

# Sets up redis to communicate results
memory = redis.StrictRedis(ip,port=6379,db=0)
memory.set('current_state','ball|||False|||False')
print "Connected to Redis"

# Load the Caffe model 
#net = cv2.dnn.readNetFromCaffe('/Users/jeff/Code/harrisonscode/AutonomousVehicle/src/python/visual/MobileNetSSD_deploy.prototxt', '/Users/jeff/Code/harrisonscode/AutonomousVehicle/src/python/visual/MobileNetSSD_deploy.caffemodel')
n = 25

# Labels of Network.
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'dining table', 12: 'dog', 13: 'horse',
    14: 'motor bike', 15: 'person', 16: 'potted plant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tv monitor' }

# Defines the state based system
states = ['default','person_features','vegetables','vehicles','signs']
default = ['person','face','car','bottle','bus','iphone','smile']
person_features = ['eye','eyeglasses','face_1','hand','left_eye','lower_body','mouth','nose','profile_face','right_eye','shoulders','smile','upper_body']
vegetables = ['apple','banana']
vehicles = ['car','bus','two_wheeler','licence_plate']
signs = ['signs','yield_sign','stop_sign','speed_sign']
past_definites = ['','','','','']
last_ball = [0,0]
pf = None

# Ball color range defined
#orangeUpper = (30,255,255) # orange ball
#orangeLower = (5,131,172)
orangeUpper = (64,255,255) # tennis ball
orangeLower = (29,86,6)
pts = deque(maxlen=30)

# Default calibrations for camera
descriptors = [['face_3',5,1.1],['eye',5,1.1],['hand',5,1.25],['smile',5,1.35],['cat_face_1',5,1.5],['lower_body',5,1.2]]

# Setup for haar detection
filenames = glob.glob('/Users/jeff/Code/harrisonscode/AutonomousVehicle/src/python/visual/haarcascade/*')
filelist = [z.split('.xml')[0].split('/')[-1] for z in filenames]

# Setup lane detection for later
#l = laneDetect()

# Create a socket object
try:
	sock1 = socket() 
	sock1.connect((ip, 5000))
except: 
	print "Cam 1 error"
	sys.exit()
try:
	sock2 = socket()
	sock2.connect((ip, 5001))
except: pass
 
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

class videorecClient():
    calibration,face = '',''
    cam_type='mono'

    def __init__(self,cam_type='mono'):
	self.cam_type = cam_type
        self.model = load_model('/Users/jeff/Code/harrisonscode/AutonomousVehicle/src/python/visual/model.pkl')
	cascade_filename='/Users/jeff/Code/harrisonscode/AutonomousVehicle/src/python/visual/haarcascade/face.xml'
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
	# Calibration
	self.calibration = StereoCalibration(input_folder='/Users/jeff/Code/harrisonscode/AutonomousVehicle/src/python/visual/calibration_data')
	print "Models loaded: Starting streaming thread"
	threading.Thread(target=self.run).start()

    def run(self):
	global last_ball
	avg = None

        while True:
	    # Init vars
	    offset, angle, sign_text = '','',''
	    combined,checked_definites,past,high_confidence = [],[],[],[]
	    current_motion,ball = [],[]
	    current_state,in_motion,upside_down = memory.get('current_state').split('|||')
	    filelist = []
	    exec('filelist = %s' % current_state)

	    # Get length of buffered data
            length1 = recvall(sock1, 16)
            if length1 == None:
		print "Cam error"
                break
	    if self.cam_type == 'stereo':
		    length2 = recvall(sock2, 16)
		    if length2 == None:
		    	print "Cam error"
		        break

	    # Process buffed data from socket into numpy
	    buf1 = recvall(sock1, int(length1))
	    if self.cam_type == 'stereo': buf2 = recvall(sock2, int(length2))
	    data1 = numpy.fromstring(buf1, dtype='uint8')
	    if self.cam_type == 'stereo': data2 = numpy.fromstring(buf2, dtype='uint8')

	    # Form numpys into frames
	    frame1 = cv2.imdecode(data1, 1)
	    if self.cam_type == 'stereo': frame2 = cv2.imdecode(data2, 1)
	   
	    # Forming images resized from frames
	    img = cv2.resize(frame1, (frame1.shape[1], frame1.shape[0]), interpolation = cv2.INTER_CUBIC)
	    if self.cam_type == 'stereo': img2 = cv2.resize(frame2, (frame2.shape[1], frame2.shape[0]), interpolation = cv2.INTER_CUBIC)
	    # Flips the img  
	    if upside_down == 'False':
		img = cv2.flip(img,0)
		if self.cam_type == 'stereo': img2 = cv2.flip(img2,1)

	    # Disparity from the images to determine distance
	    if self.cam_type == 'stereo':
		    rectified_pair = self.calibration.rectify((img, img2))
		    disp = cv2.StereoMatcher(rectified_pair[0], rectified_pair[1])

	    # Motion detect
	    if (in_motion == 'False') and (current_state == 'motion'):
		    # Preprocessing for motion detection
		    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		    gray = cv2.GaussianBlur(gray, (21,21), 0)
		    if avg is None: avg = gray.copy().astype("float")
		    cv2.accumulateWeighted(gray, avg, 0.5)
		    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
		    thresh = cv2.threshold(frameDelta, 5, 255,cv2.THRESH_BINARY)[1]
	     	    thresh = cv2.dilate(thresh, None, iterations=2)
		    _,cnts,ret, = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		    for c in cnts:
			# If the contour is too small, ignore it
			if cv2.contourArea(c) < 5000:
				continue
			(x, y, w, h) = cv2.boundingRect(c)
			c_x = ((x+(x+w))/2)*2
			c_y = ((y+(y+h))/2)*2
			if self.cam_type == 'stereo': combined.extend(['motion',str(c_x),str(c_y),'100',str(disp[c_x,c_y])])
			else: combined.extend([['motion',str(c_x),str(c_y),'100','0']])

	    # Ball Detection
	    if current_state == 'ball':
		    # Particle filtering
		    #if (pf is None) and (current_state == 'ball'):
		    #    pf = ParticleFilter(NUM_PARTICLES,(-frame1.shape[0]/2, 3*frame1.shape[0]/2),(-frame1.shape[1]/2, 3*frame1.shape[1]/2),(10, max(frame1.shape)/2))

		    #pf.elapse_time()
	    	    # Preprocessing for ball detection
		    blurred = cv2.GaussianBlur(img, (11, 11), 0)
		    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
		    mask = cv2.inRange(hsv, orangeLower, orangeUpper)
		    mask = cv2.erode(mask, None, iterations=2)
		    mask = cv2.dilate(mask, None, iterations=2)
		    #contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
		    #pf.observe(contours)
		    #x,y,radius = pf.return_most_likely(frame1)
		    #pf.resample()
		    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		    center = None
		    ball = []
		    # If there's at least one ball
		    if len(cnts) > 0:
			c = max(cnts, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
			# Only proceed if the radius meets a minimum size (distance from AV)
			if radius > 5:
				#print "RADIUS: " + str(radius)
			        if self.cam_type == 'stereo': ball=['ball',str(x),str(y),'100',str(disp[x,y])]
			        else: ball=['ball',str(center[0]),str(center[1]),str(radius),'0']
			        combined.extend([ball])
				#if self.cam_type == 'stereo': ball=['ball',str(x),str(y),'100',str(disp[x,y])]
				#else: ball=['ball',str(x),str(y),'100','0']
				'''
				# Out of bounds - resort to last ball location
				if (float(ball[1])>float(last_ball[0]+100)) or (float(ball[1])<float(last_ball[0]-100)) or (float(ball[2])>(last_ball[0]+100)) or (float(ball[2])<(last_ball[0]-100)):
					if last_ball[0] != 0:
						ball[1] = last_ball[0]
						ball[2] = last_ball[1]
				last_ball = ball[1:][:-1]
				'''

	    # Haar cascade objects
	    if (current_state == 'default') or (current_state == 'signs'):
		    # Recursive call to look for matching, selected cascades
		    for x in xrange(0, len(filelist)):
			    for i,r in enumerate(eval('self.%s' % filelist[x]).detect(img)):
				# Initialize image, coords & prediction
				filename = filelist[x]
				x0,y0,x1,y1 = r
				exec('%s = img[y0:y1, x0:x1]' % filename)
				exec('%s = cv2.cvtColor(%s,cv2.COLOR_BGR2GRAY)' % (filename,filename))
				if 'face' in filename:					
					exec('%s = cv2.resize(%s, self.model.image_size, interpolation = cv2.INTER_CUBIC)' % (filename,filename))
					exec('prediction = self.model.predict(%s)[0]' % filename)
					self.face = self.model.subject_names[prediction]
				if self.cam_type == 'stereo': combined.extend([[filename,str((x0+x1)/2),str((y0+y1)/2),'80',str(disp[(x0+x1)/2,(y0+y1)/2])]])
				else: combined.extend([[filename,str((x0+x1)/2),str((y0+y1)/2),'80','0']])
	    '''
	    # Tensorflow objects
	    if (current_state == 'default'):
		    # Resize frame to 300x300 (required by tensorflow)
		    frame_resized = cv2.resize(frame1,(300,300))
		    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
		    # Set to network the input blob 
		    net.setInput(blob)
		    # Prediction of network
		    detections = net.forward()
		    # Size of frame resize (300x300)
		    cols = frame_resized.shape[1] 
		    rows = frame_resized.shape[0]
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
			    yRightTop   = int(heightFactor * _yRightTop)
			    # Label and confidence of prediction in frame resized
			    if class_id in classNames:
				label = classNames[class_id]
				labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

				yLeftBottom = max(yLeftBottom, labelSize[1])
				xCenter = (_xLeftBottom+_xRightTop)/2
				yCenter = (_yLeftBottom+_yRightTop)/2
				if self.cam_type == 'stereo': combined.extend([[label,str(xCenter),str(yCenter),str(confidence),disp[int(widthFactor*xCenter),int(heightFactor*yCenter)]]])
				else: combined.extend([[label,str(xCenter),str(yCenter),str(confidence),'0']])
	    '''
	    checked_definites = copy(combined)
	    
	    # Inputs current objects into past 5 collection groupings
	    past_definites[0] = copy(combined)
	    for count in xrange(1,len(past_definites)): past_definites[5-count]=past_definites[4-count]
	    [past.extend(z) for z in past_definites]

	    '''
	    # High confidence detected objects are treated as foolproof
	    if combined and (isinstance(combined[0],list) == True): high_confidence = [z for z in combined if float(z[3]) > 99.1]
	    elif combined and (float(combined[3]) > 99.1): high_confidence = combined
	    for x in xrange(0,len(high_confidence)):
		# Add them to definite objects currently detected, if not already there
		if not [z for z in checked_definites if high_confidence[x][0] in checked_definites]:
			checked_definites.extend(high_confidence[x])

	    # Checks if there is correlation over time
	    for x in xrange(0, len(past)):
		names_list = [z[0] for z in past]
		for y in xrange(0,len(names_list)):
			# If there are more than 4 instances of object, and object is currently being detected
			names_check = [z for z in checked_definites if names_list[y] in z[0]]
			print str(names_check)
			if (names_list.count(names_list[y]) > 4) and (names_check) and (not [z for z in checked_definites if high_confidence[x][0] in checked_definites]):
				checked_definites.extend(names_check)

	    # Lane detection & road signs
	    if (current_state == 'sign'):
		# Gets speed limit from road signs
		if [z for z in checked_definites if 'sign' in z[0]]:
			sign_text = pytesseract.image_to_string(img)
		# Lane Detection
	        offset,angle = l.get_data(img)
	    '''
	    '''
	    # Looks for objects that overlap with any balls
	    if ball:
		x = int(ball[1])
		y = int(ball[2])
		overlap = [z for z in checked_definites if (int(z[1])>(x-30)) and (int(z[1])<(x+30)) and (int(z[2])>(y-30)) and (int(z[2])<(x+30)) and (z[0]!='ball') and (z[0] in person_features)]
	    '''

	    
	    #cv2.imshow('test',img)
	    #if cv2.waitKey(1) == 27: break
	    #print str(checked_definites)
	    # Uploads the results to AV
	    memory.set('objects_detected',str(checked_definites)+'|||'+sign_text+'|||'+offset+'|||'+angle)

	    # Says we can receive what's in the buffer next
	    sock1.send('OK')
	    if self.cam_type == 'stereo': sock2.send('OK')
