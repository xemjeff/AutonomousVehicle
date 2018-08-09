import os,threading
import redis
from time import sleep
from controls.motion import motion
from nlp.nlpEngine import nlpEngine
from sensors.gps_monitor import gps_monitor

n = nlpEngine()
m = motion()
g = gps_monitor()

# ======================== ADD THESE TO RC.LOCAL ON PI =================================
os.system('python /root/AutonomousVehicle/src/python/visual/remote/server.py &')
os.system('python /root/AutonomousVehicle/src/python/audio/serveMicrophone.py &')
os.system('pigpiod')

# Redis to get speech-to-text result & to send text-to-speech text
memory = redis.StrictRedis(host='localhost',port=6379,db=0)

# Shutdown system
def shutdown():
	memory.set('ttsOverride','My power is running low. Shutting down now. Please replace my battery')
	sleep(9)
	os.system('shutdown now')

# Checks that sensors are in nominal range
def systems_check():
	# Checks if the battery is low
	if '0x50000' in os.popen('vcgencmd get_throttled').read(): shutdown()
	# Checks if AV is far from the center gps point
	elif g.distance_from_center > 28: 
		m.turn_around()
		m.find_center()
	# Checks if the AV disconnected from the wifi
	elif 'offline' == os.popen('ping -q -w1 -c1 192.168.1.1 &>/dev/null && echo online || echo offline').read():
		m.turn_around()
		m.crawl_forward()
	sleep(10)
			
threading.Thread(target=systems_check).start()

while True:
	try:
		objects,speed_sign,tracking = memory.get('objects_detected').split('|||')
		m.process(list(objects),n.state,speed_sign,tracking)
		sleep(0.1)
	except Exception as e:
		memory.set('ttsOverride','Error: '+str(e))
		pass
