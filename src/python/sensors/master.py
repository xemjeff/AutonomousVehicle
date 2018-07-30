import os 
import redis
from time import sleep
from motion import motion
from nlp import nlp

n = nlp()
m = motion()

os.system('python /root/AutonomousVehicle/src/python/sensors/voltage_monitor &')
os.system('python /root/AutonomousVehicle/src/python/sensors/gps_monitor &')

memory = redis.StrictRedis(host='localhost',port=6379,db=0)

# Shutdown system
def shutdown():
	memory.set('ttsOverride','My power is running low. Shutting down now. Please replace my battery')
	sleep(9)
	os.system('shutdown now')

# Checks that sensors are in nominal range
def systems_check():
	if float(memory.get('voltage')) < 4750: shutdown()
	if float(memory.get('distance_from_center')) > 28: m.find_center()
	
while True:
	systems_check()
	sleep(5)
