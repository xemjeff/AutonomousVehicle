from socket import *
import cv2
import numpy
from getpass import getpass

n = 25

def recvall(conn, count):
    buf = b''
    while count:
        newbuf = conn.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)

    return buf

try:
    sock = socket() # Create a socket object
    sock2 = socket()
    sock.connect(('192.168.1.117', 5000))
    sock2.connect(('192.168.1.117', 5001))
    print 't'
    while 1:
        length = recvall(sock, 16)
        if length == None:
            break
	length2 = recvall(sock2, 16)
	if length2 == None:
	    break
        
        buf = recvall(sock, int(length))
	buf2 = recvall(sock2, int(length2))
        data = numpy.fromstring(buf, dtype='uint8')
	data2 = numpy.fromstring(buf2, dtype='uint8')

        frame1 = cv2.imdecode(data, 1)
	img = cv2.resize(frame1, (frame1.shape[1], frame1.shape[0]), interpolation = cv2.INTER_CUBIC)
	frame2 = cv2.imdecode(data2, 1)
	img2 = cv2.resize(frame2, (frame2.shape[1], frame2.shape[0]), interpolation = cv2.INTER_CUBIC)
        cv2.imshow('Client', img)
	cv2.imshow('Client2', img2)
        sock.send('OK')
	sock2.send('OK')

    sock.close()
    sock2.close()
    cv2.destroyAllWindows()
    
except Exception as e:
	print e

exit(1)
