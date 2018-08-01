import os 
import redis
from time import sleep
from controls.motion import motion
from nlp.nlpEngine import nlpEngine
from sensors.gps_monitor import gps_monitor
from sensors.voltage_monitor import voltage_monitor

n = nlpEngine()
m = motion()

# Startup audio/video services to Secondary Server
os.system('python /root/AutonomousVehicle/src/python/visual/remote/server.py &')
os.system('python /root/AutonomousVehicle/src/python/audio/serveMicrophone.py &')

# Redis to get speech-to-text result & to send text-to-speech text
memory = redis.StrictRedis(host='localhost',port=6379,db=0)

# GPS setup
g = gps_monitor()
memory.set('ttsOverride','Calibrating my center location from where I am now')
timer = 0
while True:
	if g.current_lat != 0:
		g.calibrate_center()
		memory.set('ttsOverride','Successfully calibrated to G P S. I will not wander away now')
		break
	if timer == 15: 
		memory.set('ttsOverride','G P S timeout. Calibrate me later')
		break
	sleep(1)

# Voltage setup
v = voltage_monitor()

# Vision setup
memory.set('ttsOverride','Please place an aluminum can in front of me so I can calibrate visual disparity')
memory.set('calibrate_disparity','True')

# Shutdown system
def shutdown():
	memory.set('ttsOverride','My power is running low. Shutting down now. Please replace my battery')
	sleep(9)
	os.system('shutdown now')

# Checks that sensors are in nominal range
def systems_check():
	if (v.voltage < 4750) and (v.voltage!=0): shutdown()
	elif g.distance_from_center > 28: 
		m.turn_around()
		m.find_center()
	else:
		state = n.state
		m.process(memory.get('objects'),memory.get('obstacles'),state,memory.get('speed_sign'))
	
while True:
	systems_check()
	sleep(0.1)
