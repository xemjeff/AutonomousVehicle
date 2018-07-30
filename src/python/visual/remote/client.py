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
    print '==' * n
    sock.connect(('192.168.1.3', 5000))
    
    while 1:
        length = recvall(sock, 16)
        if length == None:
            break
        
        buf = recvall(sock, int(length))
        data = numpy.fromstring(buf, dtype='uint8')

        decimg = cv2.imdecode(data, 1)
        cv2.imshow('Client', decimg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sock.send('Quit')
            break
        else:
            sock.send('OK')

    sock.close()
    cv2.destroyAllWindows()
    
except:
    pass

exit(1)
