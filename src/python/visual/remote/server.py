from socket import *
import cv2
import numpy
import netifaces as ni
from utils import *
import os,glob

try:
	ip = os.popen('nslookup raspberrypi').read().split('Address: ')[1].replace('\n','')
except:
	ip = 'localhost'
	pass

# Determines if multiple cameras exist
cams = [z for z in glob.glob('/dev/*') if z.startswith('/dev/video')]
if len(cams) == 1: multicam = False
else: multicam = True

# Transmits both camera feeds to the client
cam1 = Cam(int(cams[0][-1]))
if multicam: cam2 = Cam(int(cams[1][-1]))
cam1.start()
if multicam: cam2.start()

sock1 = socket()
sock1.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
if multicam: 
	sock2 = socket()
	sock2.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)

sock1.bind((ip,5000))
sock1.listen(True)
if multicam: 
	sock2.bind((ip,5001))
	sock2.listen(True)


conn1,addr = sock1.accept()
client1  = Client(conn1, cam1)
client1.start()
if multicam: 
	conn2,addr = sock2.accept()
	client2  = Client(conn2, cam2)
	client2.start()


