import time
import wiringpi
from time import sleep

wiringpi.wiringPiSetupGpio()

# Pan
wiringpi.pinMode(11, wiringpi.GPIO.PWM_OUTPUT)
wiringpi.pwmSetMode(wiringpi.GPIO.PWM_MODE_MS)

# Tilt
wiringpi.pinMode(15, wiringpi.GPIO.PWM_OUTPUT)
wiringpi.pwmSetMode(wiringpi.GPIO.PWM_MODE_MS)

# Speed
wiringpi.pinMode(19, wiringpi.GPIO.PWM_OUTPUT)
wiringpi.pwmSetMode(wiringpi.GPIO.PWM_MODE_MS)

# Direction
wiringpi.pinMode(13, wiringpi.GPIO.PWM_OUTPUT)
wiringpi.pwmSetMode(wiringpi.GPIO.PWM_MODE_MS)

# divide down clock
wiringpi.pwmSetClock(192)
wiringpi.pwmSetRange(2000)

# states = ['ball','croquet','sign',nlp']

class motion():
	self.direction,self.speed,self.pan,self.tilt = 0,0,0,0
	# ---------------------------------------------------------------------------
	# Key settings
	def neutral(self): wiringpi.pwmWrite(19, 150)
	def forward(self): 
		self.neutral()
		wiringpi.pwmWrite(19, 200)
	def reverse(self):
		self.neutral()
		wiringpi.pwmWrite(19, 100)
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
			num = rangeLimit(num)
			if num < 0: self.setDirectionAV(num)
	def turnLeft(self,num=0): 
		if num == 0:
			self.setDirectionAV(-0.75)
			self.forward()
			sleep(1)
			self.neutral()
		else:
			num = rangeLimit(num)
			if num > 0: self.setDirectionAV(num)
	def shiftLeft(self):
		self.setDirectionAV(-0.2)
		self.setSpeedAV(0.05)
		sleep(0.1)
		self.neutral()
	def shiftRight(self):
		self.setDirectionAV(0.2)
		self.setSpeedAV(0.05)
		sleep(0.1)
		self.neutral()
	def backup(self):
		self.reverse()
		sleep(0.4)
		self.neutral()
	#   (Pan/tilt Mount)
	def panLeft(self,num):
		num = rangeLimit(num)
		if num > 0: self.setPan(num)
	def panRight(self,num):
		num = rangeLimit(num)
		if num < 0: self.setPan(num)
	def tiltUp(self,num):
		num = rangeLimit(num)
		if num > 0: self.setTilt(num)
	def tiltDown(self,num):
		num = rangeLimit(num)
		if num < 0: self.setTilt(num)
	def shiftPanLeft(self): self.setPan(0.1)
	def shiftPanRight(self): self.setPan(-0.1)
	# Variable direction / speed (between -1 and 1)
	def setDirectionAV(self,direction):
		direction = rangeLimit(direction)
		self.direction = direction
		wiringpi.pwmWrite(13, self.toPWM(direction))
	def setSpeedAV(self,speed):
		direction = rangeLimit(speed)
		self.speed = speed
		wiringpi.pwmWrite(19, self.toPWM(speed))
	def setDirectionPan(self,direction):
		direction = rangeLimit(direction)
		self.pan = direction
		wiringpi.pwmWrite(11, self.toPWM(direction))
	def setDirectionTilt(self,direction):
		direction = rangeLimit(direction)
		self.tilt = direction
		wiringpi.pwmWrite(15, self.toPWM(direction))
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
		if self.pan == -1: self.shiftPanRight()
		elif self.pan == 1: self.shiftPanLeft()
	# Centers the object in vision
	def find_center_object(x,y,camera_width):
		self.setSpeed(0.1)
		if x < ((camera_width/2)-15): self.shiftRight()
		elif x > ((camera_width/2)+15): self.shiftLeft()
		if y < ((camera_width/2)-15): self.shiftUp()
		elif y > ((camera_width/2)+15): self.shiftDown()
		else: return 'FIXED'
	# Move self AV to center the object in view
	def track(x,y,distance,camera_width):
		# Move toward the object if centered
		if self.find_center_object(x,y,camera_width) == 'FIXED':
			self.setDirectionAV(0)			
			self.setSpeedAV(0.3)
			if distance > 0.6: sleep(distance-0.5)
			else: sleep(distance-0.2)
			self.neutral()
			return 'TRACKED'
	# Navigate toward objects of interest while avoiding obstacles
	def process(self,objects,obstacles,state,speed):
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
		# Focus only upon signs for demonstration
		elif state == 'sign':
			# Right turn sign - Turn right, wait, then circle to original
			if [z for z in objects if 'right' in z[0]]:
				self.turnRight()
				sleep(2.5)
				self.turnRight()
				self.turnRight()
				self.turnRight()
			# Left turn sign - Turn left, wait, then circle to original
			elif [z for z in objects if 'left' in z[0]]:
				self.turnLeft()
				sleep(2.5)
				self.turnLeft()
				self.turnLeft()
				self.turnLeft()
			# Yield sign - Quick start/stop
			elif [z for z in objects if 'yield' in z[0]]:
				self.forward()
				sleep(0.2)
				self.neutral()
			# Speed sign - relative speed to the sign
			elif [z for z in objects if 'speed' in z[0]]:
				if speed != 0: self.setSpeedAV(float(speed)/80)
				else: self.forward()
				sleep(2)
				self.neutral()
			# No sign, but there's supposed to be
			else: self.oob()
		# Engage with people
		elif state == 'nlp:
			self.neutral()
			# Center on the face if available
			face = [z for z in objects if 'face' in z[0]]
			if face: self.find_center_object(face[1],face[2],640)
			else:
				# Center on any motion if available
				motion = [z for z in objects if 'motion' in z[0]]
				elif motion: self.find_center_object(motion[1],motion[2],640)
			
				

		


