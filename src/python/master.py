# Setup for ball detection and tracking
# Uncomment the non-comment lines and replace 'ball' when passing data to motion class (for full features)

import os,threading
import redis
from time import sleep
from controls.motion import motion
#from nlp.nlpEngine import nlpEngine
#from sensors.gps_monitor import gps_monitor
from visual.onboard_video import onboard_video

v = onboard_video('mono')
#n = nlpEngine()
m = motion(height=240,width=320)
#g = gps_monitor()

# Serves the microphone to networked server
#os.system('python /root/AutonomousVehicle/src/python/audio/serveMicrophone.py &')
# Serves the cameras to networked server
#os.system('python /root/AutonomousVehicle/src/python/visual/remote/server.py &')

# Redis to get speech-to-text result, send text-to-speech text & detected objects visually
memory = redis.StrictRedis(os.popen('nslookup raspberrypi').read().split('Address: ')[1].replace('\n',''),port=6379,db=0)
memory.set('current_state','ball|||False')

state = 'ball'

# Shutdown system
def shutdown():
	memory.set('ttsOverride','My power is running low. Shutting down now. Please replace my battery')
	sleep(9)
	os.system('shutdown now')

# Checks that sensors are in nominal range every 10 secs.
def systems_check():
	# Checks if the battery is low
	if '0x50000' in os.popen('vcgencmd get_throttled').read(): shutdown()
	# Checks if AV is far from the center gps point
	# Uncomment for full features
	elif g.distance_from_center > 28: 
		m.turn_around()
		m.find_center()
	# Checks if the AV disconnected from the wifi
	elif 'offline' == os.popen('ping -q -w1 -c1 192.168.1.1 &>/dev/null && echo online || echo offline').read():
		m.turn_around()
		m.crawl_forward()
	sleep(10)
			
# Start the systems check thread
#threading.Thread(target=systems_check).start()

while True:
	try:
		# Sends NLP state & motion data to server
		#memory.set('current_state','ball'+'|||'+str(m.motion)+'|||'+str(m.upside_down))

		# Receive visual detection data from server and parse
		#objects,speed_sign,offset,angle = memory.get('objects_detected').split('|||')

		# Pass visual & audio information to motion class for appropriate processing
		m.process(v.ball)

		# Sleep for 1/100th of a second
		sleep(0.01)
	except Exception as e:
		memory.set('ttsOverride','Error: '+str(e))
		pass
