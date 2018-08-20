import redis
from time import sleep
from Kernel import Kernel
import threading

# TTS service
def tts(text): memory.set('ttsOverride',text)

# Redis memory (short-term, HAL)
memory = redis.StrictRedis(host='localhost', port=6379, db=0)

# AIML memory
k = Kernel()
k.saveFilepath('/root/AutonomousVehicle/src/python/nlp/memory/')
k.learn("*")

class nlpEngine():
	state = 'nlp'

	def __init__(self):
		threading.Thread(target=self.respond).start()

	def respond(self):
		while True:
			result = memory.get('stt_result')
			if result and (state=='nlp'):
				text = memory.get('stt_result')
				memory.set('stt_result','')

				# Processes meta-vars
				shiftArray(lastSaidUser,text)

				# Processes for actions in text if mode is True
				if command(text):
					tts('Okay')
				# Responds from memory
				else:
					tts(k.respond(text,'human'))		
			sleep(0.1)

