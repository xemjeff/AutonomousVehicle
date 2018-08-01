import redis
import time
import os
import re
import sys
import contextlib
import wave
import subprocess
from random import randint
from time import sleep
from datetime import datetime
from datetime import timedelta
import paramiko
import glob
import string
from shutil import copyfile
from Kernel import Kernel
import threading

#--------------------------------- SUPPORT FUNCTIONS ----------------------------------
# -------------------------------------------------------------------------------------

# Checks to see if what's said last has been said before
def checkRepeat(array):
	checkAgainst = array[0]
	sum = 0
	for count in xrange(1,len(array)):
		if (array[count] == checkAgainst) and array[count]:
			sum = sum + 1
	return sum

# Shifts array and inputs new data
def shiftArray(array, data):
        # Shifts the list by one
	for count in xrange(0,len(array)):
        	array[4 - count] = array[3 - count]
	array[0] = str(data)

# Saves something not understood to say and learn from later
def learn(string):
	global listenResponse
	tts(learn[randint(0,4)])
	listenResponse = [True, 'learn']

# Looks for command
def command(lastLine):
	global action_group
	command_count = 0
	# Split the command if there are multiple commands
	andSplit = lastLine.split(' and ')
	for b in xrange(0, len(andSplit)):
		tempArraySum = []
		wordlen = 0
		# For every word in the command
		for word in andSplit[b].split():
			if len(word) >= 3:
				# Semantic processing if syntax doesn't match
				match = filter(lambda x: '%s' % word.lower() in [z.lower() for z in re.findall('[a-zA-Z][^A-Z]*', x)], memory.keys())
				# We found a match
				if (len(match) > 0):
					wordlen += 1
					# Gets the list of all memory keys containing 'word'
					tempArrayInstance = match
					# If this is the first word of the sentence, inititalize
					if tempArraySum == []:
						tempArraySum.extend(tempArrayInstance)
					# All other words in sentence
					else:
						if len(list(set(tempArraySum).intersection(tempArrayInstance))) > 0:
							tempArraySum=list(set(tempArraySum).intersection(tempArrayInstance))
		# Activates specific action if an action group isn't being set up
		if (listenResponse[1] != 'group') and (memory.get('actionModeReq') == 'True') and (wordlen >= 2):
			# Multiple variables returned - ambiguity - go with list item with lowest length
			if len(tempArraySum) > 1:
				maxCount = 0
				maxKey = 0
				for y in xrange(0, len(tempArraySum)):
				    count = 0
				    listSplit = [z.lower() for z in re.findall('[a-zA-Z][^A-Z]*', tempArraySum[y])]
				    for x in xrange(0, len(lastLine.split())):
					    count = listSplit.count(lastLine.split()[x])
					    if count > 0:
						count += 1
				    if count > maxCount:
					    maxCount = count
					    maxKey = tempArraySum[y]
				#readableKey = " ".join(re.findall('[A-Z][a-z]*', min(tempArraySum, key=len)))
				if str(maxKey) == '0': return False
				try:
					if memory.get(maxKey).startswith('%e') == True:
						tempArraySum[0] = memory.get(maxKey)[2:]
						exec memory.get(maxKey)[2:] in globals(), locals()
						print "Executing python function"
						def subfunction():
						    return True
					else:
						readableKey = " ".join(re.findall('[A-Z][a-z]*', maxKey))
						tts(("The "+readableKey+" is "+memory.get(maxKey)))
				except: pass
			# If the sum is one, then we've found our variable to use
			elif len(tempArraySum) == 1:
			       	# Check if it's a shell command to be run - %c
				if memory.get(tempArraySum[0]).startswith('%c') == True:
					result = subprocess.check_output("%s" % memory.get(tempArraySum[0])[2:], shell=True)
		       	                memory.set('commandResult', result)
			      	        tts(result)
				       	print "Executing physical command"
				# Check if it's python function to be run - %p
				elif memory.get(tempArraySum[0]).startswith('%p') == True:
					command = memory.get(tempArraySum[0])[2:]
		       	                memory.set('roombaExec', 'exec(%s)' % command)
			       	        print "Excuting python script"
				# Check if it's a voice response - %v
				elif memory.get(tempArraySum[0]).startswith('%v') == True:
					tts(memory.get(tempArraySum[0])[2:])
			       	        print "Voice Response"
				# Check if it's a python code - %e
				elif memory.get(tempArraySum[0]).startswith('%e') == True:
					exec memory.get(tempArraySum[0])[2:] in globals(), locals()
					def subfunction():
						return True
			       	        print "Executing python command"
				# Say the variable output
				else:
					readableKey = " ".join(re.findall('[A-Z][a-z]*', tempArraySum[0]))
					tts((readableKey, memory.get(tempArraySum[0])))
			# Sleeps in case relay is activated multiple times by andSplit
			sleep(1)
		# Request to do action
		elif tempArraySum:
			memory.set('actionModeReq','True')
			return True
		# Sets action into memory
		if tempArraySum:
			#setAction(tempArraySum[0])
			if (listenResponse[1] == 'group') and (listenResponse[2] == ''):
				action_group.append('memory.get("%s")' % tempArraySum[0])
		# Indicates whether or not action was taken, to the parent function
		if tempArraySum:
			print str(tempArraySum)
			command_count += 1
	if command_count > 0:
		return True
	else:
		return False

# Yes/no/maybe/sometimes returner
def query(input,count=False):
	list = []
	names = []
	result = ''
	initial = os.popen('grep ">%s<" -A 1 /root/AutonomousVehicle/src/python/nlp/interjection_separate.aiml' % input.upper()).read()
	if initial:
		result = initial.split('"action">')[1].split('</set>')[0]
	else:
		for x in xrange(0,len(input.split())):
			if len(input.split()[x]) < 4:
				test = os.popen('grep ">%s<" -A 1 /root/AutonomousVehicle/src/python/nlp/interjection_separate.aiml' % input.split()[x].upper()).read()
			else:
				test = os.popen('grep "%s" -A 1 /root/AutonomousVehicle/src/python/nlp/interjection_separate.aiml' % input.split()[x].upper()).read()
			if test:
				list.append(test.split('"action">')[1].split('</set>')[0])
				if 'no' in test.split('"action">')[1].split('</set>')[0]: names.append(input.split()[x])
	if list and (not count): result = max(set(list),key=list.count)
	elif list and count: 
		result = ','.join(names)+"|"+str(list.count(max(set(list),key=list.count)))
	return result

# TTS service
def tts(text):
	memory.set('ttsOverride',text)

# Redis memory (short-term, HAL)
memory = redis.StrictRedis(host='localhost', port=6379, db=0)
os.system('python ../memory/setMemory.py')

# AIML memory
k = Kernel()
k.saveFilepath('/root/AutonomousVehicle/src/python/nlp/memory/')
k.learn("*")

# Local vars.
lowerLetter = lambda s: s[:1].lower() + s[1:] if s else ''
misunderstood = ["I am sorry, but I cannot understand you","What was that?","Say that again please?","I don't know what you mean","Pardon?"]
repeated = ['Why do u keep saying that?', 'Please stop saying that.', 'This is getting tiring.', 'This is getting repetitive.', 'Saying the same thing is annoying.', 'This is getting annoying.']
learn = ["What does that mean?","Pardon?","I don't know what that means","What did you mean by that?","I don't know what you meant by that"]
thanks = ["Thank you","I appreciate that","Thanks!","Interesting"]
ask = ["Can I help you?","Yes?","What is it?","How can I help you?","What do you want?"]
listenResponse = [False, '', '', '', False]
lastSaidUser = ['','','','','']
pattern,template,text = "","",""

# Startup vars. and services for speech recognition
f_temp = 200
volume = 35
furnace_override = datetime.now()

class nlpEngine():

	def __init__(self):
		threading.Thread(target=self.respond).start()

	def respond(self):
		while True:
			result = memory.get('stt_result')
			if result:
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

