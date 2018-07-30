# This class implements the AIML pattern-matching algorithm described
# by Dr. Richard Wallace at the following site:
# http://www.alicebot.org/documentation/matching.html

import marshal
import pprint
import re
import string
import sys
import glob
from random import randint

# AIML filenames for later
filelist = []
filenames = glob.glob('/root/Scripts/Alicia/memory/aiml/*')
for x in xrange(0,len(filenames)):
	filelist.append(filenames[x].split('aiml/')[1].split('.aiml')[0])

filelist = sorted(filelist, key=str.lower)

class PatternMgr:
	# special dictionary keys
	_MULTISTAR  = 6
	_TOPIC	    = 0
	_UNDERSCORE = 1
	_STAR       = 2
	_TEMPLATE   = 3
	_THAT       = 4
	_BOT_NAME   = 5
	starLen     = 0
	secondary_results = False
	
	def __init__(self):
		self._root = {}
		self._templateCount = 0
		self._botName = u"Nameless"
		punctuation = "\"`~!@#$%^&*()-_=+[{]}\|;:',<.>/?"
		self._puncStripRE = re.compile("[" + re.escape(punctuation) + "]")
		self._whitespaceRE = re.compile("\s+", re.LOCALE | re.UNICODE)

	def numTemplates(self):
		"""Return the number of templates currently stored."""
		return self._templateCount

	def setBotName(self, name):
		"""Set the name of the bot, used to match <bot name="name"> tags in
		patterns.  The name must be a single word!

		"""
		# Collapse a multi-word name into a single word
		self._botName = unicode(string.join(name.split()))

	def dump(self):
		"""Print all learned patterns, for debugging purposes."""
		pprint.pprint(self._root)

	def save(self, filename):
		"""Dump the current patterns to the file specified by filename.  To
		restore later, use restore().

		"""
		try:
			outFile = open(filename, "wb")
			marshal.dump(self._templateCount, outFile)
			marshal.dump(self._botName, outFile)
			marshal.dump(self._root, outFile)
			outFile.close()
		except Exception, e:
			print "Error saving PatternMgr to file %s:" % filename
			raise Exception, e

	def restore(self, filename):
		"""Restore a previously save()d collection of patterns."""
		try:
			inFile = open(filename, "rb")
			self._templateCount = marshal.load(inFile)
			self._botName = marshal.load(inFile)
			self._root = marshal.load(inFile)
			inFile.close()
		except Exception, e:
			print "Error restoring PatternMgr from file %s:" % filename
			raise Exception, e

	def add(self, (pattern,that,topic), template):
		"""Add a [pattern/that/topic] tuple and its corresponding template
		to the node tree.

		"""
		# TODO: make sure words contains only legal characters
		# (alphanumerics,*,_)

		# Navigate through the node tree to the template's location, adding
		# nodes if necessary.
		node = self._root
		for word in string.split(pattern):
			key = word
			if key == u"|":
				key = self._MULTISTAR
			elif key == u"_":
				key = self._UNDERSCORE
			elif key == u"*":
				key = self._STAR
			elif key == u"BOT_NAME":
				key = self._BOT_NAME
			if not node.has_key(key):
				node[key] = {}
			node = node[key]

		# navigate further down, if a non-empty "that" pattern was included
		if len(that) > 0:
			if not node.has_key(self._THAT):
				node[self._THAT] = {}
			node = node[self._THAT]
			for word in string.split(that):
				key = word
				if key == u"|":
					key = self._MULTISTAR
				elif key == u"_":
					key = self._UNDERSCORE
				elif key == u"*":
					key = self._STAR
				if not node.has_key(key):
					node[key] = {}
				node = node[key]

		# navigate yet further down, if a non-empty "topic" string was included
		if len(topic) > 0:
			if not node.has_key(self._TOPIC):
				node[self._TOPIC] = {}
			node = node[self._TOPIC]
			for word in string.split(topic):
				key = word
				if key == u"|":
					key = self._MULTISTAR
				elif key == u"_":
					key = self._UNDERSCORE
				elif key == u"*":
					key = self._STAR
				if not node.has_key(key):
					node[key] = {}
				node = node[key]


		# add the template.
		if not node.has_key(self._TEMPLATE):
			self._templateCount += 1	
		node[self._TEMPLATE] = template

	def match(self, pattern, that, topic, secondary):
		"""Return the template which is the closest match to pattern. The
		'that' parameter contains the bot's previous response. The 'topic'
		parameter contains the current topic of conversation.

		Returns None if no template is found.
		
		"""
		global secondary_results
		global filelist
		filelist = []
		filenames = glob.glob('/root/Scripts/Alicia/memory/aiml/*')
		for x in xrange(0,len(filenames)):
			filelist.append(filenames[x].split('aiml/')[1].split('.aiml')[0])
		filelist = sorted(filelist, key=str.lower)
		if secondary:
			secondary_results = True
		else:
			secondary_results = False
		if len(pattern) == 0:
			return None
		# Mutilate the input.  Remove all punctuation and convert the
		# text to all caps.
		try:
			input = string.upper(pattern)
		except:
			input = string.upper(''.join(pattern))
		input = re.sub(self._puncStripRE, " ", input)
		if that.strip() == u"": that = u"ULTRABOGUSDUMMYTHAT" # 'that' must never be empty
		thatInput = string.upper(that)
		thatInput = re.sub(self._puncStripRE, " ", thatInput)
		thatInput = re.sub(self._whitespaceRE, " ", thatInput)
		if topic.replace('*','').strip() == u"": topic = u"ATOMIC" # 'topic' must never be empty
		topicInput = string.upper(topic)
		topicInput = re.sub(self._puncStripRE, " ", topicInput)
		
		#print "input: " + str(input)
		#print "topic: " + str(topicInput)

		# Looks for match in the current topic
		patMatch, template = self._match(input.split(), thatInput.split(), topicInput.split(), self._root)
		return template
		'''
		#print "Initital: " + str(template)
		template_cache = []
		override_cache = []
		# Looks for match in all templates - catches for a template we don't want
		if ((not template) or ((not 'srai' in str(template).lower()) and (not 'xfind' in str(template).lower()) and (not "'li'" in str(template).lower()) and (not "}, u" in str(template).lower()))) and ((not '***' in str(template) or (not 'XSPLIT' in str(template)))):
			for y in xrange(0,2):
				#print "Round: " + str(y+1)
				for x in xrange(0,len(filelist)):
					#print "FileList: " + filelist[x]
					topicInput = filelist[x].upper()
					topicInput = re.sub(self._puncStripRE, " ", topicInput)
					patMatch, template = self._match(input.split()[:y], thatInput.split(), topicInput.split(), self._root)
					#print "Template: " + str(template) + "\n"
					if template and (not 'xfind' in str(template).lower()) and (not "'yours'" in str(template).lower()):
						template_cache.append(template)
				#print "Template cache: " + str(template_cache)
				if template_cache:
					#print "Found something"
					break
			if template_cache:
				return max(template_cache,key=len)
		else:
			return template
		'''

	def star(self, starType, pattern, that, topic, index):
		"""Returns a string, the portion of pattern that was matched by a *.

		The 'starType' parameter specifies which type of star to find.
		Legal values are:
		 - 'star': matches a star in the main pattern.
		 - 'thatstar': matches a star in the that pattern.
		 - 'topicstar': matches a star in the topic pattern.

		"""
		# Mutilate the input.  Remove all punctuation and convert the
		# text to all caps.
		input = string.upper(pattern)
		input = re.sub(self._puncStripRE, " ", input)
		input = re.sub(self._whitespaceRE, " ", input)
		if that.strip() == u"": that = u"ULTRABOGUSDUMMYTHAT" # 'that' must never be empty
		thatInput = string.upper(that)
		thatInput = re.sub(self._puncStripRE, " ", thatInput)
		thatInput = re.sub(self._whitespaceRE, " ", thatInput)
		if topic.strip() == u"": topic = u"ULTRABOGUSDUMMYTOPIC" # 'topic' must never be empty
		topicInput = string.upper(topic)
		topicInput = re.sub(self._puncStripRE, " ", topicInput)
		topicInput = re.sub(self._whitespaceRE, " ", topicInput)

		# Pass the input off to the recursive pattern-matcher
		patMatch, template = self._match(input.split(), thatInput.split(), topicInput.split(), self._root)
		if template == None:
			return ""

		# Extract the appropriate portion of the pattern, based on the
		# starType argument.
		words = None
		try:
			if starType == 'star':
				patMatch = patMatch[:patMatch.index(self._THAT)]
				words = input.split()
			elif starType == 'thatstar':
				patMatch = patMatch[patMatch.index(self._THAT)+1 : patMatch.index(self._TOPIC)]
				words = thatInput.split()
			elif starType == 'topicstar':
				patMatch = patMatch[patMatch.index(self._TOPIC)+1 :]
				words = topicInput.split()
			else:
				# unknown value
				raise ValueError, "starType must be in ['star', 'thatstar', 'topicstar']"
		except Exception as e:
			return ""
		
		# compare the input string to the matched pattern, word by word.
		# At the end of this loop, if foundTheRightStar is true, start and
		# end will contain the start and end indices (in "words") of
		# the substring that the desired star matched.
		foundTheRightStar = False
		start = end = j = numStars = k = 0
		for i in range(len(words)):
			# This condition is true after processing a star
			# that ISN'T the one we're looking for.
			if i < k:
				continue
			# If we're reached the end of the pattern, we're done.
			if j == len(patMatch):
				break
			if not foundTheRightStar:
				if patMatch[j] in [self._STAR, self._UNDERSCORE, self._MULTISTAR]: #we got a star
					numStars += 1
					if numStars == index:
						# This is the star we care about.
						foundTheRightStar = True
					start = i
					# Iterate through the rest of the string.
					for k in range (i, len(words)):
						# If the star is at the end of the pattern,
						# we know exactly where it ends.
						if j+1  == len (patMatch):
							end = len (words)
							break
						# If the words have started matching the
						# pattern again, the star has ended.
						if patMatch[j+1] == words[k]:
							end = k - 1
							i = k
							break
				# If we just finished processing the star we cared
				# about, we exit the loop early.
				if foundTheRightStar:
					break
			# Move to the next element of the pattern.
			j += 1

		self.starLen = len(words)
		# extract the star words from the original, unmutilated input.
		if foundTheRightStar:
			#print string.join(pattern.split()[start:end+1])
			if starType == 'star': return string.join(pattern.split()[start:end+1])
			elif starType == 'thatstar': return string.join(that.split()[start:end+1])
			elif starType == 'topicstar': return string.join(topic.split()[start:end+1])
		else: return ""

	def _returnMatchLen(self):
		return self.match_len

	def _match(self, words, thatWords, topicWords, root):
		"""Return a tuple (pat, tem) where pat is a list of nodes, starting
		at the root and leading to the matching pattern, and tem is the
		matched template.

		""" 
		# base-case: if the word list is empty, return the current node's
		# template.
		try:
			if str(pattern) == 'STOP':
				return (pattern, template)
		except Exception as e:
			pass
		if len(words) == 0:
			# we're out of words.
			pattern = []
			template = None
			if len(thatWords) > 0:
				# If thatWords isn't empty, recursively
				# pattern-match on the _THAT node with thatWords as words.
				try:
					pattern, template = self._match(thatWords, [], topicWords, root[self._THAT])
					if pattern != None:
						pattern = [self._THAT] + pattern
				except KeyError:
					pattern = []
					template = None
			elif len(topicWords) > 0:
				# If thatWords is empty and topicWords isn't, recursively pattern
				# on the _TOPIC node with topicWords as words.
				try:
					pattern, template = self._match(topicWords, [], [], root[self._TOPIC])
					if pattern != None:
						pattern = [self._TOPIC] + pattern
				except KeyError:
					pattern = []
					template = None
			if template == None:
				# we're totally out of input.  Grab the template at this node.
				pattern = []
				try: template = root[self._TEMPLATE]
				except KeyError: template = None
			return (pattern, template)

		first = words[0]
		suffix = words[1:]
		
		# If secondary results are wanted
		if secondary_results:
			# Check Special character | - these are words that will match no matter what
			if root.has_key(self._MULTISTAR):
				for x in xrange(0,len(root[self._MULTISTAR].keys())):
					for y in xrange(0, len(words)):
						if words[y] == root[self._MULTISTAR].keys()[x]:
							if "'li'" in str(root[self._MULTISTAR][words[y]]):
								# Build a list of responses and return a random one
								lilist = str(root[self._MULTISTAR][words[y]]).split("'li'")
								ranlist = []
								for z in xrange(1, len(lilist)):
									try:
										ranlist.append(lilist[z].split("u'")[1].split("'")[0])
									except Exception as e:
										ranlist.append(lilist[z].split('u"')[1].split('"')[0])
								return ('STOP', "***"+ranlist[randint(0,len(ranlist)-1)])
							else:
								response = str(root[self._MULTISTAR][words[y]]).split("u'")[1].split("'")
								if response[0] == 'action':
									return ('STOP', "***action"+str(root[self._MULTISTAR][words[y]]).split(", u'")[-1].split("'")[0])
								else:
									return ('STOP', "***"+response[0])

		# Check underscore
		if root.has_key(self._UNDERSCORE):
			# Must include the case where suf is [] in order to handle the case
			# where a * or _ is at the end of the pattern.
			for j in range(len(suffix)+1):
				suf = suffix[j:]
				pattern, template = self._match(suf, thatWords, topicWords, root[self._UNDERSCORE])
				if template is not None:
					newPattern = [self._UNDERSCORE] + pattern
					return (newPattern, template)

		# Check first
		if root.has_key(first):
			pattern, template = self._match(suffix, thatWords, topicWords, root[first])
			if template is not None:
				newPattern = [first] + pattern
				return (newPattern, template)

		# check bot name
		if root.has_key(self._BOT_NAME) and first == self._botName:
			pattern, template = self._match(suffix, thatWords, topicWords, root[self._BOT_NAME])
			if template is not None:
				newPattern = [first] + pattern
				return (newPattern, template)
	
		# check star
		if root.has_key(self._STAR):
			# Must include the case where suf is [] in order to handle the case
			# where a * or _ is at the end of the pattern.
			for j in range(len(suffix)+1):
				suf = suffix[j:]
				pattern, template = self._match(suf, thatWords, topicWords, root[self._STAR])
				if template is not None:
					newPattern = [self._STAR] + pattern
					return (newPattern, template)

		# No matches were found.
		return (None, None)			
