from __future__ import print_function
import gi,sys,redis,os
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
GObject.threads_init()
Gst.init(None)

# Connect to the rPi
print "Waiting for the rPis IP"
while True:
	try:
		ip = os.popen('nslookup raspberrypi').read().split('Address: ')[1].replace('\n','')
	except:
		sleep(1)
print "Aquired IP "+str(ip)

memory = redis.StrictRedis(host=ip,port=6379,db=0)


pipeline = Gst.parse_launch('uridecodebin name=source ! audioconvert !' +
                            ' audioresample ! pocketsphinx name=asr !' +
                            ' fakesink')
source = pipeline.get_by_name('source')
bus = pipeline.get_bus()

# Start playing
pipeline.set_state(Gst.State.PLAYING)
while True:
    msg = bus.timed_pop(Gst.CLOCK_TIME_NONE)
    if msg:
        if msg.type == Gst.MessageType.EOS:
            break
        struct = msg.get_structure()
        if struct and struct.get_name() == 'pocketsphinx':
	    # Send results back to the rPi
            if struct['final']:
                memory.set('stt_result',struct['final'])
            elif struct['hypothesis']:
                memory.set('stt_result',struct['hypothesis'])
                pass

# Free resources
pipeline.set_state(Gst.State.NULL)
