# Not fun for this particular hardware, to get it into the right mode
# Try holding it for as long as you can, until the green light keeps flashing
# Make sure to start pigpiod before
from ping_monitor import ping_monitor
from time import sleep
import pigpio,redis

# To tell people to move
memory = redis.StrictRedis(host='localhost',port=6379,db=0)

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

# Init vars.
past_ball = ['','','','','','','','','','','','','','','']

class motion():
	direction,speed,pan,tilt = 0,0,0,0
	pan_direction = 'right'
	motion = False
	offset = 0 # Calibrate by getting the 'tracking' var when in center of lane
	# ---------------------------------------------------------------------------
	# Key settings
	def neutral(self): 
		pi.set_servo_pulsewidth(SERVO[0],1500)
		self.motion = False
	def forward(self): 
		self.neutral()
		self.setSpeedAV(1)
		self.motion = True
	def reverse(self):
		self.neutral()
		self.setSpeedAV(-0.4)
		self.motion = True
	# User relatable settings
	#   (Pan/Tilt Servos)
	def nodYes(self):
		original_x,original_y = self.pan,self.tilt
		pi.set_servo_pulsewidth(SERVO[3],800)
		pi.set_servo_pulsewidth(SERVO[2],1200)
		sleep(0.2)
		pi.set_servo_pulsewidth(SERVO[2],1500)
		sleep(0.2)
		pi.set_servo_pulsewidth(SERVO[2],1200)
		sleep(0.2)
		pi.set_servo_pulsewidth(SERVO[2],1500)
		sleep(0.2)
		pi.set_servo_pulsewidth(SERVO[2],1200)
		self.pan,self.tilt = original_x,original_y
	def shakeNo(self):
		original_x,original_y = self.pan,self.tilt
		pi.set_servo_pulsewidth(SERVO[3],800)
		pi.set_servo_pulsewidth(SERVO[2],1200)
		sleep(0.2)
		pi.set_servo_pulsewidth(SERVO[3],1000)
		sleep(0.2)
		pi.set_servo_pulsewidth(SERVO[3],600)
		sleep(0.2)
		pi.set_servo_pulsewidth(SERVO[3],1000)
		sleep(0.2)
		pi.set_servo_pulsewidth(SERVO[3],800)
		self.pan,self.tilt = original_x,original_y
	def defaultHead(self):
		pi.set_servo_pulsewidth(SERVO[3],800)
		pi.set_servo_pulsewidth(SERVO[2],1200)
		self.tilt = 1200
		self.pan = 800
	# User relatable settings
	#   (AV Motors)
	def turnRight(self): 
		self.setDirectionAV(-0.75)
		self.forward()
		sleep(0.5)
		self.neutral()
	def turnRight(self,num=0): 
		if num == 0:
			self.setDirectionAV(0.75)
			self.forward()
			sleep(0.5)
			self.neutral() 
		else:
			num = self.rangeLimit(num)
			if num < 0: self.setDirectionAV(num)
	def turnLeft(self,num=0): 
		if num == 0:
			self.setDirectionAV(-0.75)
			self.forward()
			sleep(1)
			self.neutral()
		else:
			num = self.rangeLimit(num)
			if num > 0: self.setDirectionAV(num)
	def shiftLeft(self):
		self.setDirectionAV(0.75)
		self.setSpeedAV(0.4)
		sleep(0.15)
		self.neutral()
	def shiftRight(self):
		self.setDirectionAV(-0.75)
		self.setSpeedAV(0.4)
		sleep(0.15)
		self.neutral()
	def backup(self):
		self.reverse()
		sleep(0.4)
		self.neutral()
	def ahead(self):
		self.forward()
		sleep(0.4)
		self.neutral()
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
	def shiftPanLeft(self): self.setDirectionPan(self.pan+0.05)
	def shiftPanRight(self): self.setDirectionPan(self.pan-0.05)
	def shiftTiltUp(self): self.setDirectionTilt(self.tilt+0.05)
	def shiftTiltDown(self): self.setDirectionTilt(self.tilt-0.05)
	# Variable direction / speed (between -1 and 1)
	def setDirectionAV(self,direction):
		direction = self.rangeLimit(direction)
		self.direction = direction
		pi.set_servo_pulsewidth(SERVO[1],int(self.toPWM(direction)))
	def setSpeedAV(self,speed):
		direction = self.rangeLimit(speed)
		self.speed = speed
		pi.set_servo_pulsewidth(SERVO[0],int(self.toPWM(speed)))
	def setDirectionPan(self,direction):
		direction = self.rangeLimit(direction)
		self.pan = direction
		pi.set_servo_pulsewidth(SERVO[3],int(self.toPWM(direction)))
	def setDirectionTilt(self,direction):
		direction = self.rangeLimit(direction)
		self.tilt = direction
		pi.set_servo_pulsewidth(SERVO[2],int(self.toPWM(direction)))
	# Support functions
	def rangeLimit(self,num):
		if num > 1: num = 1
		elif num < -1: num = -1
		return num
	def toPWM(self,num):
		return 1500 + num*(2000-1500)
	# ---------------------------------------------------------------------------
	# Finds the object when out of sight
	def oob(self):
		# Return tilt to proper position 
		if (self.tilt != 800) or (self.pan != 1200): self.defaultHead()
		# Obstacle is near AV - stop and wait for it to pass
		if p.ping <= 12:
			self.neutral()
			sleep(1)
		# Wait is over - turn left
		elif self.motion == False:
			self.shiftLeft()
		''' The code below is for the head pan/tilt search to be used with IMU
		if (self.pan_direction == 'right') and (self.pan != -1):
			self.shiftPanRight()
		elif (self.pan_direction == 'right') and (self.pan == -1):
			self.pan_direction = 'left'
			self.shiftPanLeft()
		elif (self.pan_direction == 'left') and (self.pan != 1):
			self.shiftPanRight()
		elif (self.pan_direction == 'left') and (self.pan == 1):
			self.pan_direction = 'right'
			self.shiftPanRight()
		'''
	# Centers the object in vision
	def find_center_object(str(x),str(y),camera_width=640):
		self.setSpeed(0.1)
		if x < ((camera_width/2)-15): self.shiftRight()
		elif x > ((camera_width/2)+15): self.shiftLeft()
		if y < ((camera_width/2)-15): self.shiftTiltUp()
		elif y > ((camera_width/2)+15): self.shiftTiltDown()
		else: return 'FIXED'
	# Move self AV to center the object in view
	def track(str(x),str(y),distance,camera_width=640):
		excused = False
		# Move toward the object if centered
		if self.find_center_object(x,y,camera_width) == 'FIXED':
			self.setDirectionAV(0)			
			self.forward()
			# While loop to make sure there aren't objects in front of AV from p.ping
			time_to_target = (distance/12)*1.2 # 1.2 being AV speed in feet per second
			timer = 0
			while (distance > 12) and (timer < time_to_target):
				# Obstacle is near AV - stop and wait for it to pass
				if p.ping <= 24:
					if excused == False: memory.set('ttsOverride',"Pardon me. Trying to get by")
					self.neutral()
					sleep(1)
				# Wait is over - go forward
				elif self.motion == False:
					self.forward()
					sleep(0.5)
					timer += 0.5
				# Nothing in the way - proceed
				else:
					sleep(0.5)
					timer += 0.5
			self.neutral()
			return 'TRACKED'
	# Navigate toward objects of interest while avoiding obstacles
	def process(self,objects,state,speed,tracking,motion):
		# Focus upon tracking the ball
		if state == 'ball':
			ball = [z for z in objects if 'ball' in z[0]]
			if ball: self.track(ball[1],ball[2],ball[4],640)
			else: self.oob()
		# Focus upon croquet hoops for demonstration
		elif state == 'croquet':
			croquet = [z for z in objects if 'croquet' in z[0]]
			# Track and move to the next hoop
			if croquet: 
				# Pass through the hoop to find the next
				if self.track(croquet[1],croquet[2],croquet[4],640) == 'TRACKED':
					self.forward()
					sleep(1)
					self.neutral()
			# Relocated the next hoop
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
				sleep(2)
				self.ahead()
			# Speed sign
			elif [z for z in objects if 'speed' in z[0]]:
				self.ahead()
			# No sign, but there's supposed to be
			elif state == 'sign': self.oob()
			# Lane detection
			elif state == 'lane':
				# If there's a lane
				if (tracking<-10) or (tracking>3):
					# Turn right
					if tracking < self.offset:
						if self.direction < 0: self.direction = 0
						self.direction += 0.05
					# Turn left
					elif tracking > self.offset:
						if self.direction > 0: self.direction = 0
						self.direction += -0.05
					self.setDirectionAV(self.direction)
					self.ahead()
				# No lane, find one
				else: self.oob()
		# Engage with people
		elif state == 'nlp':
			self.neutral()
			# Center on the face if available
			face = [z for z in objects if 'face' in z[0]]
			if face: self.find_center_object(face[1],face[2],640)
			else:
				# Center on any motion if available
				motion = [z for z in objects if 'motion' in z[0]]
				if motion: self.find_center_object(motion[1],motion[2],640)
			
				

		


