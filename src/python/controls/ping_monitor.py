from time import sleep
import RPi.GPIO as GPIO
import threading,time

# GPIO Mode (BOARD / BCM)
GPIO.setmode(GPIO.BCM)
 
# Set GPIO Pins
GPIO_TRIGGER = 24
GPIO_ECHO = 23

# Set GPIO direction (IN / OUT)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)
 
def distance():
    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
    StartTime = time.time()
    StopTime = time.time()
    while GPIO.input(GPIO_ECHO) == 0:
        StartTime = time.time()
    while GPIO.input(GPIO_ECHO) == 1:
        StopTime = time.time()
    TimeElapsed = StopTime - StartTime
    distance = (TimeElapsed * 34300) / 2
    return float(distance*0.394)

class ping_monitor():

	def __init__(self):
		self.ping = 0
		threading.Thread(target=self._get_data).start()

	def _get_data(self):
		while True:
			self.ping = distance()
			sleep(0.01)
