import os,sys

try:
	ip = os.popen('Secondary').read().split('Address: ')[1].replace('\n','')
	os.system("gst-launch-1.0 audiotestsrc ! audio/x-raw, endianness=1234, signed=true, width=16, depth=16, rate=44100, channels=1, format=S16LE ! tcpclientsink host=%s port=3000" % ip)
except:
	print "The Secondary Server is offline"
	sys.exit () 
