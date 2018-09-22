# Not fun for this particular hardware, to get it into the right mode
# Try holding it for as long as you can, until the green light keeps flashing
# Make sure to start pigpiod before
from ping_monitor import ping_monitor
from random import randint
from time import sleep
import pigpio,redis,threading,os
import ast,sys

# To tell people to move
memory = redis.StrictRedis(host='localhost',port=6379,db=0)
excused = ["Pardon me, but you're in my way","Do you mind if I get by?","Oops. My mistake","Sorry","Excuse me","Coming through here!"]

SERVO = [19,13,20,21]     

p = ping_monitor()
pi = pigpio.pi()

# Initialize the servos
for x in SERVO:
	if x != 19:
		pi.set_mode(x, pigpio.OUTPUT)
		pi.set_servo_pulsewidth(x, 1500)

# Initialize the esc
pi.set_mode(SERVO[0], pigpio.OUTPUT)
pi.set_servo_pulsewidth(SERVO[0],0)

# Tilt min-800,max-2500
# Pan min-500,max-2500

# Init vars.
degree = float(1.25/90)

class motion():
	direction,speed,pan,tilt,height,width,low_height,offset = 0,0,0,0,0,0,0,0
	low_margin_min_x,low_margin_max_x,high_margin_min_x,high_margin_max_x = 0,0,0,0
	low_margin_min_y,low_margin_max_y,high_margin_min_y,high_margin_max_y = 0,0,0,0
	pan_direction = 'right'
	last_state = ''
	object_history = []
	moving_while_tracking,motion,force_stop,tracked,upside_down = False,False,False,False,False
	object_history = ['','','','','','','','','','']
	offset = 0 # Calibrate by getting the 'tracking' var when in center of lane

	# Set initial positions and start watch-dog thread
        def __init__(self,height,width):
		self.height,self.width = height,width
		self.low_margin_min_x = ((self.width/2)-(self.width/35))
		self.low_margin_max_x = ((self.width/2)+(self.width/35))
		self.high_margin_min_x = ((self.width/2)-(self.width/18))
		self.high_margin_max_x = ((self.width/2)+(self.width/18))
		self.low_margin_min_y = ((self.height/2)-(self.height/35))
		self.low_margin_max_y = ((self.height/2)+(self.height/35))
		self.high_margin_min_y = ((self.height/2)-(self.height/18))
		self.high_margin_max_y = ((self.height/2)+(self.height/18))
		self.low_height = self.height/9
		self.defaultHead()
		#threading.Thread(target=self.run).start()

	# ---------------------------------------------------------------------------
	# Key settings
	def neutral(self): 
		pi.set_servo_pulsewidth(SERVO[0],1500)
		# If cam is flipped - return to normal
		if self.tilt == 1: self.setDirectionTilt(-0.25)
		# Stop any audio
		try: os.system('killall mplayer')
		except: pass
		self.motion = False
	def forward(self):
		self.force_stop = False
		if self.pan != -0.75: self.defaultHead()
		self.setSpeedAV(0.3) 
		self.motion = True
	def reverse(self):
		# Special sequence set by Jeff
		self.setSpeedAV(-0.1)
		sleep(0.05)
		self.neutral()
		sleep(0.05)
		self.setSpeedAV(-0.3)
		# Flip the camera
		self.setDirectionTilt(1)
		# Start the truck beeping
		os.popen('mplayer /root/AutonomousVehicle/src/media/reverse.wav &')
		self.motion = True
	# User relatable settings
	#   (Pan/Tilt Servos)
	def nodYes(self):
		self.motion = True
		pi.set_servo_pulsewidth(SERVO[2],1200)
		sleep(0.2)
		pi.set_servo_pulsewidth(SERVO[2],1500)
		sleep(0.2)
		pi.set_servo_pulsewidth(SERVO[2],1200)
		sleep(0.2)
		pi.set_servo_pulsewidth(SERVO[2],1500)
		sleep(0.2)
		pi.set_servo_pulsewidth(SERVO[2],1200)
		sleep(0.2)
		pi.set_servo_pulsewidth(SERVO[2],self.toServoPWM(self.tilt))
		self.motion = False
	def shakeNo(self):
		self.motion = True
		pi.set_servo_pulsewidth(SERVO[3],self.toServoPWM(self.pan-0.2))
		sleep(0.2)
		pi.set_servo_pulsewidth(SERVO[3],self.toServoPWM(self.pan+0.2))
		sleep(0.2)
		pi.set_servo_pulsewidth(SERVO[3],self.toServoPWM(self.pan-0.2))
		sleep(0.2)
		pi.set_servo_pulsewidth(SERVO[3],self.toServoPWM(self.pan+0.2))
		sleep(0.2)
		pi.set_servo_pulsewidth(SERVO[3],self.toServoPWM(self.pan))
		self.motion = False
	def defaultHead(self):
		self.setDirectionPan(-0.75)
		self.setDirectionTilt(-0.25)
	# User relatable settings
	#   (AV Motors)
	def turnLeft(self,num=90): 
		# Standard AV motion
		if self.last_state != 'lane': self.setDirectionAV(0.75)
		# Advanced AV motion
		else:
			# If there's an offset
			if self.offset != 0:
				# Presumed too far left
				if self.offset > 0: self.setDirectionAV(self.direction-(self.offset*.1))
				# Presumed too far right
				else: self.setDirectionAV(self.direction+(self.offset*.1))
			else: self.setDirectionAV(0.75)
		self.forward()
		sleep(degree*num)
		self.neutral() 
	def turnRight(self,num=90): 
		# Standard AV motion
		if self.last_state != 'lane': self.setDirectionAV(-0.75)
		# Advanced AV motion
		else:
			# If there's an offset
			if self.offset != 0:
				# Presumed too far left
				if self.offset > 0: self.setDirectionAV(self.direction-(self.offset*.1))
				# Presumed too far right
				else: self.setDirectionAV(self.direction+(self.offset*.1))
			else: self.setDirectionAV(-0.75)
		self.forward()
		sleep(degree*num)
		self.neutral() 
	def backup(self):
		self.reverse()
		sleep(0.2)
		self.neutral()
	def ahead(self):
		self.forward()
		sleep(0.4)
		self.neutral()
	def sidestep(self,direction='left'):
		if direction == 'left': p_1,p_2= -0.75,0.75
		else: p_1,p_2 = 0.75,-0.75
		self.setDirectionAV(p_1)
		self.forward()
		if self.check_ping(0.75):
			self.setDirectionAV(p_2)
			if self.check_ping(0.75):
				self.setDirectionAV(p_1)
				if self.check_ping(0.2):
					self.neutral()
					self.setDirectionAV(0)
					return True
		else:
			self.reverse()
			sleep(0.5)
			self.neutral()
			self.setDirectionAV(0)	
			return False
		self.setDirectionAV(0)
		return False
	#   (Pan/tilt Mount)
	def panLeft(self,num):
		num = self.rangeLimit(num)
		if num > 0: self.setDirectionPan(num)
	def panRight(self,num):
		num = self.rangeLimit(num)
		if num < 0: self.setDirectionPan(num)
	def tiltUp(self,num):
		num = self.rangeLimit(num)
		if num > 0: self.setDirectionTilt(num)
	def tiltDown(self,num):
		num = self.rangeLimit(num)
		if num < 0: self.setDirectionTilt(num)
	def shiftPanLeft(self): self.setDirectionPan(self.pan+0.01)
	def shiftPanRight(self): self.setDirectionPan(self.pan-0.01)
	def shiftTiltUp(self): self.setDirectionTilt(self.tilt+0.01)
	def shiftTiltDown(self): self.setDirectionTilt(self.tilt-0.01)
	# Variable direction / speed (between -1 and 1)
	def setDirectionAV(self,direction):
		direction = self.rangeLimit(direction)
		self.direction = direction
		self.motion = True
		pi.set_servo_pulsewidth(SERVO[1],int(self.toPWM(direction)))
		self.motion = False
	def setSpeedAV(self,speed):
		direction = self.rangeLimit(speed)
		self.speed = speed
		self.motion = True
		pi.set_servo_pulsewidth(SERVO[0],int(self.toPWM(speed)))
		self.motion = False
	def setDirectionPan(self,direction):
		direction = self.rangeLimit(direction)
		self.pan = direction
		self.motion = True
		pi.set_servo_pulsewidth(SERVO[3],int(self.toServoPWM(direction)))
		self.motion = False
		sleep(0.01)
	def setDirectionTilt(self,direction):
		direction = self.rangeLimit(direction)
		self.tilt = direction
		self.motion = True
		pi.set_servo_pulsewidth(SERVO[2],int(self.toServoPWM(direction)))
		self.motion = False
		sleep(0.01)
	# Support functions
	def rangeLimit(self,num):
		if num > 1: num = 1
		elif num < -1: num = -1
		return num
	def toPWM(self,num):
		return 1500 + num*(2000-1500)
	def toServoPWM(self,num):
		return 1500 + num*(2500-1500)
	def check_ping(self,time):
		timer = 0
		while timer <= time:
			if p.ping <= 8: 
				self.neutral()	
				return False
			timer += 0.05
			sleep(0.05)
		return True
	# Stops/starts the vehicle on its own when conditions are met
	def run(self):
		while True:
			if (p.ping <= 8) and self.motion: self.neutral()
			'''
			# If it's the first time the AV is too close to an object - and isn't final position
			if (p.ping <= 18) and (self.force_stop == False) and (self.tracked == False) and self.motion: 
				self.neutral()
				self.backup()
				self.force_stop = True
				# Tries to sidestep around obstacle before asking to be excused
				#if not self.sidestep('right'):
				#	if not self.sidestep('left'):
				memory.set('ttsOverride',excused[randint(0,5)])
			# For when vehicle recently stopped from too close - and is tracking an object
			elif (self.moving_while_tracking == True) and (p.ping > 18) and self.force_stop:
				# If stopped while tracking, and in final position, backup a bit
				if self.tracked:
					self.backup()
					self.moving_while_tracking = False
					self.force_stop = False
				# If stopped while tracking, but path is now free, go forward again
				else:
					self.forward()
					self.force_stop = False
			'''
			# Checks if camera is upside down to send signal
			#if self.tilt >= 0.3: self.upside_down = True
			#else: self.upside_down = False
			sleep(0.05)
	# ---------------------------------------------------------------------------
	# Alternative to an IMU
	def moveOOB(self):
		diff = ((-0.75-self.pan)*-1)*100
		if diff > 0: self.turnLeft(diff)
		elif diff < 0: self.turnRight(diff*-1)
		self.setDirectionPan(-0.75)
		self.setDirectionAV(0)
	# Finds the object when out of sight
	def oob(self):
		# Stops if moving and lost object
		if self.motion: self.neutral()
		# Memory relative position predictions
		if self.object_history[0]:
			# Looks for rapid y-axis movement - look up
			if (self.object_history[0][1] > self.height-50): self.setDirectionTilt(self.tilt+0.2)
			# Looks for last position to the right - turns EVEN further right
			elif (self.object_history[0][0] > self.width-50): self.setDirectionAV(self.direction-0.4)
			# Looks for last position to the left - turns EVEN further left
			elif (self.object_history[0][0] < 50): self.setDirectionAV(self.direction+0.4)
		# Standard pan / tilt back and forth
		else:
			# Negate images directly UP		
			if (self.tilt > 0) and (self.tilt < 0.4): self.setDirectionTilt(0.4)
			# Reset the tilt
			elif self.tilt > 0.7: self.setDirectionTilt(-0.5)
			# Memory relative position predictions
			if self.object_history[0]:
				# Looks for rapid y-axis movement - turns head back
				if (self.object_history[0][1] > self.height-50): self.setDirectionTilt(0.9)
				# Looks for last position to the right - turns further right
				elif (self.object_history[0][0] > self.width-50): self.setDirectionPan(self.tilt-0.2)
				# Looks for last position to the left - turns further left
				elif (self.object_history[0][0] < 50): self.setDirectionPan(self.tilt+0.2)
				self.object_history[0] = ''
				sleep(0.7)
			# Pans back and forth, incrementing up the tilt each time it turns from right to left
			if (self.pan_direction == 'right') and (self.pan != -1): 
				self.shiftPanRight()
			elif (self.pan_direction == 'right') and (self.pan == -1):
				self.pan_direction = 'left'
				self.shiftPanLeft()
				self.setDirectionTilt(self.tilt+0.1)
			elif (self.pan_direction == 'left') and (self.pan != 1):
				self.shiftPanLeft()
			elif (self.pan_direction == 'left') and (self.pan == 1):
				self.pan_direction = 'right'
				self.shiftPanRight()
				
	# Centers the object in vision
	def find_center_object(self,x,y,distance):
		# Past distances comparison
		#if [z for z in self.object_history if z != '']:
		#	distances = [z[2] for z in self.object_history if len(z) == 4][:3]
		#	delta = max(abs(x-y) for (x,y) in zip(distances[1:],distances[:-1]))
		#else: delta = 29
		# If pan isn't in default, move AV to direction pan is pointed to
		#if (self.pan != -0.75) and (delta > 30): self.moveOOB()
		# Low Margin of Error for stationary tracking
		if self.motion == False:
			if (y > self.low_margin_max_y) or (y < self.low_margin_min_y) or (x > self.low_margin_max_x) or (x < self.low_margin_min_x):
				if y > self.low_margin_max_y: self.shiftTiltDown()
				elif y < self.low_margin_min_y: self.shiftTiltUp()
				if x > self.low_margin_max_x: self.shiftPanLeft()
				elif x < self.low_margin_min_x: self.shiftPanRight()
				else: return 'FIXED'
			else: return 'FIXED'
		# High Margin of Error for On-the-go tracking
		elif self.motion == True:
			if (y > self.high_margin_max_y) or (y < self.high_margin_min_y) or (x > self.high_margin_max_x) or (x < self.high_margin_min_x):
				if y > self.high_margin_max_y: self.shiftTiltDown()
				elif y < self.high_margin_min_y: self.shiftTiltUp()
				if x > self.high_margin_max_x: self.setDirectionAV(self.direction+0.05)
				elif x < self.high_margin_min_x: self.setDirectionAV(self.direction-0.05)
				else: return 'FIXED'	
			else: return 'FIXED'
			# Too close - back up a bit
			#if (y < self.low_height) and (self.tilt < -0.95): self.backup()
	# Center & Move AV to the Object
	def track(self,x,y,distance):
		# Move toward the object if centered
		if (self.find_center_object(x,y,distance) == 'FIXED') and (p.ping > 10):
			if (self.pan != -0.75): self.moveOOB()
			else: self.setDirectionAV(0)			
			self.ahead()
			self.moving_while_tracking = True
			self.tracked = False
		# Do nothing if object is already tracked
		elif (self.find_center_object(x,y,distance) == 'FIXED') and (distance >= 70):
			self.neutral()
			self.moving_while_tracking = False
			self.tracked = True
			return "TRACKED"
		# Default 
		else:
			self.tracked = False
	# Navigate & interact with objects based on the self-State via NLP
	def process(self,objects,state='ball'):
		objects = ast.literal_eval(objects)
		# Input object into object history
		for count in xrange(1,len(self.object_history)): self.object_history[10-count]=self.object_history[9-count]
		if objects: self.object_history[0] = [float(objects[0][1]),float(objects[0][2]),float(objects[0][3])]
		else: self.object_history[0] = []
		# Focus upon tracking the ball
		if state == 'ball':
			#print str(self.object_history)
			ball = [z for z in objects if 'ball' in z[0]]
			if ball: 
				ball = ball[0]
				self.track(float(ball[1]),float(ball[2]),float(ball[3]))
			else: self.oob()
		# Sign obeyance and lane detection
		elif (state == 'sign') or (state == 'lane'):
			# Right turn sign - Turn right, wait, then circle to original
			if [z for z in objects if 'right' in z[0]]:
				self.turnRight()
			# Left turn sign - Turn left, wait, then circle to original
			elif [z for z in objects if 'left' in z[0]]:
				self.turnLeft()
			# Stop Sign or Yield
			elif [z for z in objects if ('stop' in z[0]) or ('yield' in z[0])]:
				self.neutral()
				while p.ping <= 6: sleep(0.5)
				self.forward()
			# Speed sign
			elif [z for z in objects if 'speed' in z[0]]:
				if speed != 0: self.setSpeedAV(speed*0.1)
			# No sign, but there's supposed to be
			elif state == 'sign': self.oob()
			# Lane detection
			elif state == 'lane':
				# If there's a lane
				if (offset < -10) or (offset > 3):
					# Turn right
					if offset < self.offset:
						if self.direction < 0: self.direction = 0
						self.direction += 0.05
					# Turn left
					elif offset > self.offset:
						if self.direction > 0: self.direction = 0
						self.direction += -0.05
					self.setDirectionAV(self.direction)
					self.ahead()
				# No lane, find one
				else: self.oob()
		# Engage with people
		elif state == 'nlp':
			self.neutral()
			hand = [z for z in objects if 'hand' in z[0]]
			face = [z for z in objects if 'face' in z[0]]
			if face: self.find_center_object(face[1],face[2])
			elif hand:
				# Input object into object history
				for count in xrange(1,len(self.object_history)): self.object_history[10-count]=self.object_history[9-count]
				self.object_history[0] = [hand[1],hand[2],hand[4]]
				x_axis = [z[0] for z in self.object_history]
				delta = max(abs(x-y) for (x,y) in zip(x_axis[1:],x_axis[:-1]))
				# If someone is waving - beep
				if delta > 150: 
					os.system('/root/AutonomousVehicle/src/media/beep.wav')
					self.object_history = ['','','','','','','','','','']


