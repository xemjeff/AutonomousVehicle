from socket import *
import cv2
import numpy
import netifaces as ni
#from models import *
from utils import *
import os

# Transmits both camera feeds to the client
cam1 = Cam(0)
cam2 = Cam(2)
cam1.start()
cam2.start()

sock1 = socket()
sock1.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
sock2 = socket()
sock2.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
ip = 'localhost'

sock1.bind((ip,5000))
sock1.listen(True)
sock2.bind((ip,5001))
sock2.listen(True)

while 1:
        conn1,addr = sock1.accept()
        client1  = Client(conn1, cam1)
        client1.start()
        conn2,addr = sock2.accept()
        client2  = Client(conn2, cam2)
        client2.start()

cam1.stop()     
cam2.stop()          
sock1.close()
sock2.close()
