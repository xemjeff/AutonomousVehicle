import glob,logging,cv2,Image,os,sys,pytesseract,itertools,redis,threading,imutils,numpy
from socket import *
from getpass import getpass
from matplotlib import pyplot as plt
from helper.common import *
from helper.video import *
from lane_detection.line_fit_frames import laneDetect
# add facerec to system path
sys.path.append("/root/AutonomousVehicle/src/python/visual/facerec/py/")
from facerec.model import PredictableModel
from facerec.feature import Fisherfaces
from facerec.distance import EuclideanDistance
from facerec.classifier import NearestNeighbor
from facerec.validation import KFoldCrossValidation
from facerec.serialization import save_model, load_model
from facedet.detector import CascadedDetector
from time import sleep
from collections import deque
from stereovision.calibration import StereoCalibration

# Connect to the rPi
print "Waiting for the rPis IP"
while True:
	try:
		ip = os.popen('nslookup raspberrypi').read().split('Address: ')[1].replace('\n','')
	except:
		sleep(1)
print "Aquired IP "+str(ip)

# Sets up redis to communicate results
memory = redis.StrictRedis(ip,port=6379,db=0)
memory.set('current_state','ball|||False')
print "Connected to Redis"

# Load the Caffe model 
net = cv2.dnn.readNetFromCaffe('/root/AutonomousVehicle/src/python/visual/MobileNetSSD_deploy.prototxt', '/root/AutonomousVehicle/src/python/visual/MobileNetSSD_deploy.caffemodel')
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

# Ball color range defined
orangeUpper = (30,255,255)
orangeLower = (5,131,172)
pts = deque(maxlen=30)

# Default calibrations for camera
descriptors = [['face_3',5,1.1],['eye',5,1.1],['hand',5,1.25],['smile',5,1.35],['cat_face_1',5,1.5],['lower_body',5,1.2]]

# Setup for haar detection
filenames = glob.glob('/root/AutonomousVehicle/src/python/visual/haarcascade/*')
filelist = [z.split('.xml')[0].split('/')[-1] for z in filenames]

# Setup lane detection for later
l = laneDetect()

# Create a socket object
sock1 = socket() 
sock2 = socket()
sock1.connect((ip, 5000))
sock2.connect((ip, 5001))

# Disparity
block_matcher = cv2.StereoBM()
 
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
    calibration = ''
    face = ''

    def __init__(self):
        self.model = load_model('/root/AutonomousVehicle/src/python/visual/model.pkl')
	cascade_filename='/root/AutonomousVehicle/src/python/visual/haarcascade/face.xml'
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
	self.calibration = StereoCalibration(input_folder='/root/AutonomousVehicle/src/python/visual/calibration_data')
	print "Models loaded: Starting streaming thread"
	threading.Thread(target=self.run).start()

    def run(self):
	avg = None

        while True:
	    # Init vars
	    tracking, sign_text = '',''
	    combined, checked_definites = [],[]
	    current_motion,ball = [],[]
	    current_state,in_motion = memory.get('current_state').split()
	    filelist = []
	    exec('filelist = %s' % current_state)

	    # Get length of buffered data
            length1 = recvall(sock1, 16)
            if length1 == None:
                break
            length2 = recvall(sock2, 16)
            if length2 == None:
                break

	    # Process buffed data from socket into numpy
	    buf1 = recvall(sock1, int(length1))
	    buf2 = recvall(sock2, int(length2))
	    data1 = numpy.fromstring(buf1, dtype='uint8')
	    data2 = numpy.fromstring(buf2, dtype='uint8')

	    # Form numpys into frames
	    frame1 = cv2.imdecode(data1, 1)
	    frame2 = cv2.imdecode(data2, 1)
	    
	    # Forming images resized from frames
	    img = cv2.resize(frame1, (frame1.shape[1], frame1.shape[0]), interpolation = cv2.INTER_CUBIC)
	    img2 = cv2.resize(frame2, (frame2.shape[1], frame2.shape[0]), interpolation = cv2.INTER_CUBIC)

	    # Disparity from the images to determine distance
	    rectified_pair = calibration.rectify((img, img2))
	    disparity = block_matcher.compute(rectified_pair[0], rectified_pair[1], disptype=cv2.CV_32F)
	    norm_coeff = 255 / disparity.max()
	    disp = disparity * norm_coeff / 255

	    # Motion detect
	    if in_motion == 'False':
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
			combined.extend(['motion',str(c_x),str(c_y),'100',disp[c_x,c_y]])

	    # Ball Detection
	    if current_state == 'ball':
	    	    # Preprocessing for ball detection
		    blurred = cv2.GaussianBlur(frame1, (11, 11), 0)
		    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
		    mask = cv2.inRange(hsv, orangeLower, orangeUpper)
		    mask = cv2.erode(mask, None, iterations=2)
		    mask = cv2.dilate(mask, None, iterations=2)
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
			if radius > 10:
				ball=['ball',str(x),str(y),'100',disp[x,y]]
				# Out of bounds - resort to last ball location
				if (int(ball[1])>(last_ball[0]+100)) or (int(ball[1])<(last_ball[0]-100)) or (int(ball[2])>(last_ball[0]+100)) or (int(ball[2])<(last_ball[0]-100)):
					ball[1] = last_ball[0]
					ball[2] = last_ball[1]
				last_ball = ball[1:][:-1]
			combined.extend(ball)

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
				combined.extend([[filename,str((x0+x1)/2),str((y0+y1)/2),'80',disp[(x0+x1)/2,(y0+y1)/2]]])

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
				yCenter = (_yLeftBottom+_yRightTop)/2)
				combined.extend([[label,str(xCenter),str(yCenter),str(confidence),disp[xCenter,yCenter]]])

	    # Inputs current objects into past 5 collection groupings
	    past_definites[0] = definites
	    for count in xrange(1,len(past_definites)): past_definites[5-count]=past_definites[4-count]
	    [past.extend(z) for z in past_definites]

	    # High confidence detected objects are treated as foolproof
	    high_confidence = [z for z in combined if float(z[3]) > 99.1]
	    for x in xrange(0,len(high_confidence)):
		# Add them to definite objects currently detected, if not already there
		if not [z for z in checked_definites if high_confidence[x][0] in checked_definites]:
			checked_definites.extend(high_confidence[x])

	    # Checks if there is correlation over time
	    for x in xrange(0, len(past)):
		names_list = [z[0] for z in past]
		for y in xrange(0,len(names_list)):
			# If there are more than 4 instances of object, and object is currently being detected
			names_check = [z for z in definites if names_list[y] in z[0]]
			if (names_list.count(names_list[y]) > 4) and (names_check) and (not [z for z in checked_definites if high_confidence[x][0] in checked_definites]):
				checked_definites.extend(names_check)

	    # Lane detection & road signs
	    if (current_state == 'sign'):
		# Gets speed limit from road signs
		if [z for z in checked_definites if 'sign' in z[0]]:
			sign_text = pytesseract.image_to_string(img)
		# Lane Detection
	        tracking = l.get_data(img)

	    # Uploads the results to AV
	    memory.set('objects_detected',str(checked_definites)+'|||'+sign_text+'|||'+tracking)

	    # Get what's in the buffer next
	    sock1.send('OK')
	    sock2.send('OK')
