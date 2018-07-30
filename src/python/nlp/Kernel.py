# -*- coding: latin-1 -*-
print "Starting the Kernel (~40 sec in Total)"
"""This file contains the public interface to the aiml module."""
import AimlParser
import DefaultSubs
import Utils
from PatternMgr import PatternMgr
from WordSub import WordSub
from unidecode import unidecode
from ConfigParser import ConfigParser
import copy
import os
import pint
import random
import re
import string
import sys
import time
import threading
import xml.sax
import nltk
import ast
import pickle
import glob
import redis
import requests
import itertools
from datetime import datetime
from random import randint
from nltk.tag import pos_tag
from nltk.stem.snowball import SnowballStemmer
from text2math import text2math
from collections import defaultdict
from collections import OrderedDict
from profanity import profanity 
from string import punctuation
from random import shuffle

# Redis to relay system messages
memory = redis.StrictRedis(host='localhost', port=6379, db=0)

# Checks if there's a number in string
def hasNumber(s):
    return any(i.isdigit() for i in s)

me = ['i','me','my','mine','myself']
you = ['you','your','yours','yourself']
wish = ['want', 'like', 'get', 'wish', 'find', 'locate', 'where']
first_wordQ = ['what','when','why','where','who','how','are','any','which','can','does','do','if','is']
self_stance = ["i","me","my","mine","myself"]
you_stance = ["you","your","yours","yourself"]
others_stance = ["he","guy","girl","boy","woman","she","him","her","his","himself","herself","them","they","their","those"]
posKind = ["What other kinds of %s do you like?", "Are there any other %s that you like?", "Why do you like %s?", "Do you like any other %s?"]
negKind = ["What other kinds of %s you don't like?", "Are there any other %s that you dislike?", "Why don't you like %s?", "Do you not like any other %s?"]
ask = ["Can I help you?","Yes?","What is it?","How can I help you?","What do you want?"]
code_words = ['codes','scripts']
code_regex = re.compile('|'.join(map(re.escape, code_words)))
denied = ["I'm sorry, but you do not have access","Access Denied","You are not authorized to access this computer"]
misunderstood = ["I am sorry, but I cannot understand you","What was that?","Say that again please?","I don't know what you mean","Pardon?"]
repeated = ['Why do you keep saying that?', 'Please stop saying that.', 'This is getting tiring.', 'This is getting repetitive.', 'Saying the same thing is annoying.', 'This is getting annoying.']
learn = ["What does that mean?","Pardon? Please explain that.","I don't know what that means. Could you elaborate?","What did you mean by that?","I don't know what you meant by that. Please explain yourself."]
thanks = ["Thank you.","I appreciate that.","Thanks!","Interesting..."]
nevermind = ["Nevermind...", "I was asking you a question, but forget about it.", "That wasn't helpful.", "Not helpful in the slightest."]
blocked = ["I hate it when you feel %s.", "Stop feeling %s, okay?", "You are too %s for me.", "I wish you weren't feeling %s."]
unknown = ["I don't know.","I'm not sure.","I don't know that yet.","Who knows?"]
profanities = ['Please stop with the vularities?', 'I would prefer not to reply to such filth.', 'Please cease swearing.', 'I would appreciate you not swearing.']
filler = ["so...","you know","like,","like...","right...","kind of","actually","i mean","sort of","basically","believe me","or something","perfect","cool","literally","pretty much","um","as it were"]
listenResponse = [False, '', '', '', False]
restricted = ['##ml-ot','##AGI','##linux','#python']
lastSaidUser,lastSaidBot = ['1','2','3','4','5'],['1','2','3','4','5']
pattern = ""
template = ""
global_name = ''
last_url = ''
interjection = ['?','!','.']
quiet = True

# Descriptions
attrib = [[["length"],['short','shorter','smaller','lesser','less','little'],['long','longer','larger','greater','more','bigger','big']], [["age"],['younger','young'],['older']], [["temperature"],['colder','cooler','cold','lower'],['higher','hotter','hot','warmer','warm']]]

# Unit characteristics
characteristic = {
    "in" : "inch",
    "ft" : "feet",
    "yd" : "yard",
    "mi" : "mile",
    "nm" : "nanometer",
    "mm" : "millimeter",
    "cm" : "centimeter",
    "m" : "meter",
    "km" : "kilometer",
    "f"  : "delta_degF",
    "c"  : "delta_degC",
    "fahrenheit" : "delta_degF",
    "celsius" : "delta_degC",
    "k" : "degK",
    "kelvin" : "degK",
    "lb" : "pound",
    "lbs" : "pounds",
    "t" : "ton",
    "oz" : "ounce",
    "pt" : "pint",
    "qt" : "quart",
    "gal" : "gallon",
    "ml" : "milliliter",
    "l" : "liter",
    "p" : "micrometer"
}

desc = {
	"geo": "a country",
	"org": "a company",
	"per": "a person",
	"gpe": "a type of politics",
	"tim": "a time",
	"art": "an experience",
	"eve": "something that happened",
	"nat": "about the natural world"
}

filelist = []

# Capitalize all sentences in string
def uppercase(matchobj):
    return matchobj.group(0).upper()

def capitalize(s):
    return re.sub('^([a-z])|[\.|\?|\!]\s*([a-z])|\s+([a-z])(?=\.)', uppercase, s)

# Yes/no/maybe/sometimes returner
def query(input,count=False):
	input = input.strip(string.punctuation)
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

class Kernel:    
    # module constants
    _globalSessionID = "_global" # key of the global session (duh)
    _maxHistorySize = 100 # maximum length of the _inputs and _responses lists
    _maxRecursionDepth = 100 # maximum number of recursive <srai>/<sr> tags before the response is aborted.
    # special predicate keys
    _inputHistory = "_inputHistory"     # keys to a queue (list) of recent user input
    _outputHistory = "_outputHistory"   # keys to a queue (list) of recent responses.
    _inputStack = "_inputStack"         # Should always be empty in between calls to respond()
    sentiment = 0
    element = ""
    folderpath = ""
    predicateSet = False

    def __init__(self):
	self._verboseMode = False
        self._version = "AIML v2. NLP"
        self._brain = PatternMgr()
	self.match_len = 0
        self._respondLock = threading.RLock()
        self._textEncoding = "utf-8"

        # set up the sessions        
        self._sessions = {}
        self._addSession(self._globalSessionID)

        # Set up the bot predicates
        self._botPredicates = {}
        #self.setBotPredicate("name", "Nameless")
	self.setBotPredicate("name","autonomous vehicle")
	self.setBotPredicate("age","a few months in the making")
	self.setBotPredicate("gender","female")
	self.setBotPredicate("like","to drive around")
	self.setBotPredicate("etype","AI")
	self.setBotPredicate("order","AI")
	self.setBotPredicate("location","the United States")
	self.setBotPredicate("nationality","the United States")
	self.setBotPredicate("state","vermont")
	self.setBotPredicate("country","the USA")
	self.setBotPredicate("website","I'm an independent being")
	self.setBotPredicate("botmaster","programmers")
	self.setBotPredicate("master","my programmers")
	self.setBotPredicate("friends","everyone i meet")
	self.setBotPredicate("feelings","calm")
	self.setBotPredicate("emotions","curious")

        # set up the word substitutors (subbers):    I CHANGED DEFAULTPERSON to DEFAULTPERSON2
        self._subbers = {}
        self._subbers['gender'] = WordSub(DefaultSubs.defaultGender)
        self._subbers['person'] = WordSub(DefaultSubs.defaultPerson)
        self._subbers['person2'] = WordSub(DefaultSubs.defaultPerson2)
        self._subbers['normal'] = WordSub(DefaultSubs.defaultNormal)
	self._subbers['desc'] = WordSub(desc)
        
        # set up the element processors
        self._elementProcessors = {
            "bot":          self._processBot,
            "condition":    self._processCondition,
	    "command": 	    self._processCommand,
            "date":         self._processDate,
            "formal":       self._processFormal,
            "gender":       self._processGender,
            "get":          self._processGet,
            "gossip":       self._processGossip,
            "id":           self._processId,
            "input":        self._processInput,
            "javascript":   self._processJavascript,
            "learn":        self._processLearn,
            "li":           self._processLi,
            "lowercase":    self._processLowercase,
            "person":       self._processPerson,
            "person2":      self._processPerson2,
            "random":       self._processRandom,
            "text":         self._processText,
            "sentence":     self._processSentence,
            "set":          self._processSet,
            "size":         self._processSize,
            "sr":           self._processSr,
            "srai":         self._processSrai,
            "star":         self._processStar,
            "system":       self._processSystem,
            "template":     self._processTemplate,
            "that":         self._processThat,
            "thatstar":     self._processThatstar,
            "think":        self._processThink,
            "topicstar":    self._processTopicstar,
            "uppercase":    self._processUppercase,
            "version":      self._processVersion,
        }

    def bootstrap(self, brainFile = None, learnFiles = [], commands = []):
        """Prepare a Kernel object for use.

        If a brainFile argument is provided, the Kernel attempts to
        load the brain at the specified filename.

        If learnFiles is provided, the Kernel attempts to load the
        specified AIML files.

        Finally, each of the input strings in the commands list is
        passed to respond().

        """
        start = time.clock()
        if brainFile:
            self.loadBrain(brainFile)

        # learnFiles might be a string, in which case it should be
        # turned into a single-element list.
        learns = learnFiles
        try: learns = [ learnFiles + "" ]
        except: pass
        for file in learns:
            self.learn(file)
            
        # ditto for commands
        cmds = commands
        try: cmds = [ commands + "" ]
        except: pass
        for cmd in cmds:
            print self._respond(cmd, self._globalSessionID)
            
        if self._verboseMode:
            print "Kernel bootstrap completed in %.2f seconds" % (time.clock() - start)

    def verbose(self, isVerbose = True):
        """Enable/disable verbose output mode."""
        self._verboseMode = isVerbose

    def version(self):
        """Return the Kernel's version string."""
        return self._version

    def numCategories(self):
        """Return the number of categories the Kernel has learned."""
        # there's a one-to-one mapping between templates and categories
        return self._brain.numTemplates()

    def resetBrain(self):
        """Reset the brain to its initial state.

        This is essentially equivilant to:
            del(kern)
            kern = aiml.Kernel()

        """
        del(self._brain)
        self.__init__()

    def loadBrain(self, filename):
        """Attempt to load a previously-saved 'brain' from the
        specified filename.

        NOTE: the current contents of the 'brain' will be discarded!

        """
        if self._verboseMode: print "Loading brain from %s..." % filename,
        start = time.clock()
        self._brain.restore(filename)
        if self._verboseMode:
            end = time.clock() - start
            print "done (%d categories in %.2f seconds)" % (self._brain.numTemplates(), end)

    def saveBrain(self, filename):
        """Dump the contents of the bot's brain to a file on disk."""
        if self._verboseMode: print "Saving brain to %s..." % filename,
        start = time.clock()
        self._brain.save(filename)
        if self._verboseMode:
            print "done (%.2f seconds)" % (time.clock() - start)

    def getPredicate(self, name, sessionID = _globalSessionID):
        """Retrieve the current value of the predicate 'name' from the
        specified session.

        string is returned.

        """
        try: return self._sessions[sessionID][name]
        except KeyError: return ""

    def setPredicate(self, name, value, sessionID = _globalSessionID):
        """Set the value of the predicate 'name' in the specified
        session.

        If sessionID is not a valid session, it will be created. If
        name is not a valid predicate in the session, it will be
        created.

        """
        self._addSession(sessionID) # add the session, if it doesn't already exist.
	if type(value) is not list:
		self._sessions[sessionID][name] = value.strip().lower()

    def resetPredicates(self, sessionID = _globalSessionID):
	self.setPredicate("name","Friend",sessionID)
	self.setPredicate('question_num','0',sessionID)
	self.setPredicate('action','',sessionID)
	self.setPredicate('mean','',sessionID)
	self.setPredicate('personality','unknown',sessionID)
	self.setPredicate('topic','greetings',sessionID)
	self.setPredicate('topic_list','',sessionID)
	self.setPredicate('eContent','0',sessionID)
	self.setPredicate("age","",sessionID)
	self.setPredicate("gender","",sessionID)
	self.setPredicate("like","",sessionID)
	self.setPredicate("location","",sessionID)
	self.setPredicate("state","",sessionID)
	self.setPredicate("city","",sessionID)
	self.setPredicate("email","",sessionID)
	self.setPredicate("country","",sessionID)
	self.setPredicate("website","",sessionID)
	self.setPredicate("family","",sessionID)
	self.setPredicate("friends","",sessionID)
	self.setPredicate("birthday","",sessionID)
	self.setPredicate("ethics","",sessionID)
	self.setPredicate("feelings","",sessionID)
	self.setPredicate("emotions","",sessionID)
	self.setPredicate("religion","",sessionID)
	self.setPredicate("birthplace","",sessionID)
	self.setPredicate("talkabout","",sessionID)
	self.setPredicate("going","",sessionID)
	self.setPredicate("favoritecolor","",sessionID)
	self.setPredicate("favoriteband","",sessionID)
	self.setPredicate("favoritemovie","",sessionID)
	self.setPredicate("favoriteactor","",sessionID)
	self.setPredicate("favoritefood","",sessionID)
	self.setPredicate("favoritebook","",sessionID)
	self.setPredicate("favoriteauthor","",sessionID)
	self.setPredicate("favoritesubject","",sessionID)
	self.setPredicate("favoritesong","",sessionID)
	self.setPredicate("favoriteactress","",sessionID)
	self.setPredicate("favoritephilosopher","",sessionID)
	self.setPredicate("favoritequestion", "",sessionID)
	self.setPredicate("favoriteseason","",sessionID)
	self.setPredicate("favoritesport","",sessionID)

    def getBotPredicate(self, name):
        """Retrieve the value of the specified bot predicate.

        If name is not a valid bot predicate, the empty string is returned.        

        """
        try: return self._botPredicates[name]
        except KeyError: return ""

    def setBotPredicate(self, name, value):
        """Set the value of the specified bot predicate.

        If name is not a valid bot predicate, it will be created.

        """
        self._botPredicates[name] = value
        # Clumsy hack: if updating the bot name, we must update the
        # name in the brain as well
        if name == "name":
            self._brain.setBotName(self.getBotPredicate("name"))

    def setTextEncoding(self, encoding):
        """Set the text encoding used when loading AIML files (Latin-1, UTF-8, etc.)."""
        self._textEncoding = encoding

    def loadSubs(self, filename):
        """Load a substitutions file.

        The file must be in the Windows-style INI format (see the
        standard ConfigParser module docs for information on this
        format).  Each section of the file is loaded into its own
        substituter.

        """
	try:
		inFile = file(filename)
		parser = ConfigParser()
		parser.readfp(inFile, filename)
		inFile.close()
		for s in parser.sections():
		    # Add a new WordSub instance for this section.  If one already
		    # exists, delete it.
		    if self._subbers.has_key(s):
		        del(self._subbers[s])
		    self._subbers[s] = WordSub()
		    # iterate over the key,value pairs and add them to the subber
		    for k,v in parser.items(s):
		        self._subbers[s][k] = v
	except Exception as e:
		pass

    def _addSession(self, sessionID):
        """Create a new session with the specified ID string."""
        if self._sessions.has_key(sessionID):
            return
        # Create the session.
        self._sessions[sessionID] = {
            # Initialize the special reserved predicates
            self._inputHistory: [],
            self._outputHistory: [],
            self._inputStack: []
        }
	self.resetPredicates(sessionID)
        
    def _deleteSession(self, sessionID):
        """Delete the specified session."""
        if self._sessions.has_key(sessionID):
            _sessions.pop(sessionID)

    def getSessionData(self, sessionID = None):
        """Return a copy of the session data dictionary for the
        specified session.

        If no sessionID is specified, return a dictionary containing
        *all* of the individual session dictionaries.

        """
        s = None
        if sessionID is not None:
            try: s = self._sessions[sessionID]
            except KeyError: s = {}
        else:
            s = self._sessions
        return copy.deepcopy(s)

    # Returns list with all words that match to a predicate, with their corresponding predicates
    def matchSessionPredicates(self, input, sessionID = _globalSessionID):
	input = input.lower().strip(string.punctuation)
        try: s = self._sessions[sessionID]
        except KeyError: s = {}
        predicates = copy.deepcopy(s).keys()
	try: test = sorted([z for z in input.split() if (z in predicates)],key=len)[::-1]
	except: test = ''
	if test: return test
	else: 
		listing = []
		for x in xrange(0, len(input.split())):
			try: test = predicates[[z.startswith(input.split()[x]) for z in predicates].index(True)]
			except: pass
			if test: listing.append(test)
	return sorted(listing,key=len)[::-1]

    def saveSessionData(self):
	return self._sessions

    def loadSessionData(self, data):
	self._sessions = data #ast.literal_eval(data)

    def learn(self, filename):
        """Load and learn the contents of the specified AIML file.

        If filename includes wildcard characters, all matching files
        will be loaded and learned.

        """
        for f in glob.glob(self.folderpath+filename):
            if self._verboseMode: print "Loading %s..." % f,
            start = time.clock()
            # Load and parse the AIML file.
            parser = AimlParser.create_parser()
            handler = parser.getContentHandler()
            handler.setEncoding(self._textEncoding)
            try: parser.parse(f)
            except xml.sax.SAXParseException, msg:
                err = "\nFATAL PARSE ERROR in file %s:\n%s\n" % (f,msg)
                sys.stderr.write(err)
                continue
            # store the pattern/template pairs in the PatternMgr.
            for key,tem in handler.categories.items():
                self._brain.add(key,tem)
            # Parsing was successful.
            if self._verboseMode:
                print "done (%.2f seconds)" % (time.clock() - start)

    # gets the element data for updateAIML
    def getElement(self):
        try: return str(self.element)
        except UnicodeError: return "Get Element Error"

    # Saves filename for everything
    def saveFilepath(self,filepath):
	global filelist
	self.folderpath = filepath
	return

    # Returns a 'normal' subbing
    def normalize(self,text):
	return self._subbers['normal'].sub(text)

    # saves a response to output history
    def save(self, input, sessionID = _globalSessionID):
        """Return the Kernel's response to the input string."""
        if len(input) == 0:
            return ""

        #ensure that input is a unicode string
        try: input = input.decode(self._textEncoding, 'replace')
        except UnicodeError: pass
        except AttributeError: pass
        
        # Add the session, if it doesn't already exist
        self._addSession(sessionID)

        # add the data from this exchange to the history lists
        outputHistory = self.getPredicate(self._outputHistory, sessionID)
        outputHistory.append(input)
        
        try: return input.encode(self._textEncoding)
        except UnicodeError: return "Save Error"

    # getPredicate function to get the action predicate of response, but doesn't change history or action predicate
    def getAction(self, input, sessionID = _globalSessionID):
        """Return the Kernel's response to the input string."""
        if len(str(input)) == 0:
            return ""
	
        #ensure that input is a unicode string
        try: input = input.decode(self._textEncoding, 'replace')
        except UnicodeError: pass
        except AttributeError: pass
        
        # prevent other threads from stomping all over us.
        self._respondLock.acquire()

        # Add the session, if it doesn't already exist
        self._addSession(sessionID)

	pAction = self.getPredicate('action',sessionID)
	self.setPredicate('action','',sessionID)
        # split the input into discrete sentences
        sentences = Utils.sentences(str(input))
        finalResponse = ""
        for s in sentences:
            # Fetch the response
            finalResponse += self._respond(s, sessionID)
        
        # release the lock and return
        self._respondLock.release()
	if not finalResponse.strip():
		action = 'not_understand'
	else:
		action = ''

	self.setPredicate('action',pAction,sessionID)
        #self._sessions[sessionID]['action'] = pAction
        try: return unidecode(action)
        except KeyError: return ""

    # respond function, but doesn't change history or action predicate
    def hypothetical(self, input, sessionID = _globalSessionID):
        """Return the Kernel's response to the input string."""
        if len(input) == 0:
            return ""

        #ensure that input is a unicode string
        try: input = input.decode(self._textEncoding, 'replace')
        except UnicodeError: pass
        except AttributeError: pass
        
        # prevent other threads from stomping all over us.
        self._respondLock.acquire()

        # Add the session, if it doesn't already exist
        self._addSession(sessionID)
	pAction = self.getPredicate('action',sessionID)
        # split the input into discrete sentences
        sentences = Utils.sentences(input)
        finalResponse = ""
        for s in sentences:
            # Fetch the response
            response = self._respond(s, sessionID)

            # append this response to the final response.
            finalResponse += (response + "  ")
        finalResponse = finalResponse.strip()
        
        # release the lock and return
        self._respondLock.release()
	self._sessions[sessionID]['action'] = pAction
        try: return finalResponse.encode(self._textEncoding)
        except UnicodeError: return finalResponse

    def respond(self, input, sessionID = _globalSessionID, focus="ON"):
        """Return the Kernel's response to the input string."""
	# Catch bad input
	try:
		if len(unidecode(input.strip()).strip(string.punctuation)) == 0:
			ques = ask[randint(0,4)]
			self.save(ques,sessionID)
			return ques
	except: return ''

	# Check for profanity
	if (profanity.contains_profanity(input)) and (not 'sex' in input.lower()) and (len(input.split()) < 4) and (not 'sex' in self.folderpath):
		prof = profanities[randint(0,3)]
		self.save(prof,sessionID)
		return prof

        # Ensure that input is a unicode string
        try: input = input.decode(self._textEncoding, 'replace')
        except UnicodeError: pass
        except AttributeError: pass
	input = self._subbers['normal'].sub(input)
	try:
		input = input.decode('utf-8')
		input = unidecode(input)
		input = re.sub( r'([a-zA-Z])([0-9])', r'\1 \2', input)
	except:
		print "Unicode error"
		return 'Unicode error'

        # prevent other threads from stomping all over us.
        self._respondLock.acquire()

        # Add the session, if it doesn't already exist
        self._addSession(sessionID)

	# Resets predicate
	finalResponse = ""

	# Filters out filler words
	filler_words = [x for x in filler if ((len(x.split()) == 1) and (x in input.split())) or ((len(x.split()) > 1) and (x in input))]
	for x in xrange(0,len(filler_words)):
		input = input.replace(filler_words[x],'')
	if not input.strip(string.punctuation).strip():
		return ''

	# Guard against double negatives
	try:
		test = query(input,count=True)
		if test and (test.split('|')[0]) and (not int(test.split('|')[1]) & 1): input = ' '.join([z for z in t.split() if not z in test.split('|')[0].split(',')])
	except: pass

	# Gets the POS constituents
	tokens = nltk.word_tokenize(input)
	tagged = nltk.pos_tag(tokens)
	nouns = [s[0].lower() for s in tagged if ('NN' in s[1][:2]) and ((s != input.split()[0]) or (len(s[1]) == 3))]

	# Count phrasing vars in sentence
	self_count, you_count, others_count = 0,0,0
	input_test = unidecode(input.lower()).strip(string.punctuation)
	for x in xrange(0, len(input_test.split())):
		if input_test.split()[x] in self_stance:
			self_count += 1
		elif input_test.split()[x] in you_stance:
			you_count += 1
		elif input_test.split()[x] in others_stance:
			others_count += 1

	# Derives question from text AND allows for dictionary search override
	question_word = [s for s in first_wordQ if s in input.split()[0].lower().split("'")[0]]

	# User Predicates Search if talking about self OR you, but not about others
	if (self_count or you_count) and (not (self_count and you_count)) and (not others_count):
		try:
			# Test for predicate requests from user
			predicate_list = self.matchSessionPredicates(input,sessionID)
			result = ''
			if predicate_list:
				predicate = ''
				for x in xrange(0,len(predicate_list)):
					if (not 'you' in predicate_list[x]):
						predicate = predicate_list[x]
						if ('favorite' in predicate_list[x]):
							predicate = "favorite"+str(input.lower().split('favorite')[1].split()[0])
						if (question_word or ('?' in input)):
							# Retrieves user predicate
							if ('my' in input.lower()) or ('i' in input.lower()):
								finalResponse += " "+str(self.getPredicate(predicate,sessionID))+". "
								self.setTopic(predicate,sessionID)
								self.setBotPredicate("lastuserpredicate",self.getBotPredicate(predicate))
								self.predicateSet = True
							# Retrieves self predicate
							elif 'you' in input.lower():
								finalResponse += " "+str(self.getBotPredicate(predicate))+". "
								self.setTopic(predicate,sessionID)
								self.setBotPredicate("lastuserpredicate",self.getBotPredicate(predicate))
								self.predicateSet = True
						else: 
							# Sets user predicate if found in passive search
							if ('my' in input.lower()) or ('i' in input.lower()) and (not you_count) and (not others_count):
								tokens = nltk.word_tokenize(input.replace(predicate,''))
								tagged = nltk.pos_tag(tokens)
								test = [seq[0] for seq in tagged if (seq[1] in {'JJ','CD'}) or ('NN' in seq[1]) or ('VB' in seq[1])]
								if test:
									self.setPredicate(predicate,test[-1],sessionID)
									self.setTopic(predicate,sessionID)
									self.setBotPredicate("lastuserpredicate",self.getBotPredicate(predicate))
									self.predicateSet = True
						break
		except:
			pass

	# --- Main Memory Search ---
	if (not finalResponse):
		# Default Processing
		sentences = Utils.sentences(input)
		response = ''
		# Do a basic search of AIML
		for s in sentences:
			# Add the input to the history list before fetching the
			response = self._respond(s, sessionID)
			if response and (not '+++' in response): finalResponse += " "+response+"."

	# Admit no knowledge
	if (not finalResponse): finalResponse = "I don't know"
	else: 
		try:
			test_topic = n.getEntity(finalResponse)
			if test_topic and (test_topic != 'ERROR'):
				for x in xrange(0,len(test_topic)):
					self.setBotPredicate('topic_'+test_topic[x][1],test_topic[x][0])
		except: pass

	# Further parsing
	finalResponse = '.'.join(list(OrderedDict.fromkeys(finalResponse.split("."))))
	finalResponse = capitalize(finalResponse.strip().replace('  ',' ').replace(' ?','?').replace(' .','.').replace('..','.').replace('?.','?').replace('!.','!').replace("i'm","I'm"))
	# Removes duplicate adjacent words
	finalResponse = ' '.join([k for k, g in itertools.groupby(finalResponse.split())])
	# Makes sure there is only one question asked at a time
	question_split = re.split('[?]',finalResponse)
	if len(question_split) >= 3:
		sentences = n.getSentences(finalResponse)
		response = ' '.join([z for z in sentences if not '?' in z])
		finalResponse = response+' '+max([z for z in sentences if '?' in z],key=len)			

	# If response is being repeated
	try:
		if finalResponse in self.getPredicate(self._outputHistory,sessionID)[-1]: finalResponse = repeated[randint(0,len(repeated)-1)]
	except: pass

	# add the input to history list
	inputHistory = self.getPredicate(self._inputHistory, sessionID)
	inputHistory.append(input)
	while len(inputHistory) > self._maxHistorySize:
		inputHistory.pop(0)
	self.setPredicate(self._inputHistory, inputHistory, sessionID)

	# add the data from this exchange to the history lists
	outputHistory = self.getPredicate(self._outputHistory, sessionID)
	outputHistory.append(finalResponse)
	while len(outputHistory) > self._maxHistorySize:
		outputHistory.pop(0)
	self.setPredicate(self._outputHistory, outputHistory, sessionID)

	# ========== END OF TRUST ===========

        # release the lock and return
        self._respondLock.release()

        try: return finalResponse.encode(self._textEncoding)
        except UnicodeError: return finalResponse

    # This version of _respond() just fetches the response for some input.
    # It does not mess with the input and output histories.  Recursive calls
    # to respond() spawned from tags like <srai> should call this function
    # instead of respond().
    def _respond(self, input, sessionID):
        """Private version of respond(), does the real work."""
        if len(input) == 0:
            return ""

        # guard against infinite recursion
        inputStack = self.getPredicate(self._inputStack, sessionID)
	#print str(inputStack)
        if len(inputStack) > self._maxRecursionDepth:
            if self._verboseMode:
                err = "WARNING: maximum recursion depth exceeded (input='%s')" % input.encode(self._textEncoding, 'replace')
                sys.stderr.write(err)
            return ""

        # push the input onto the input stack
        inputStack = self.getPredicate(self._inputStack, sessionID)
        inputStack.append(input)
        self.setPredicate(self._inputStack, inputStack, sessionID)

        # fetch the bot's previous response, to pass to the match()
        # function as 'that'.
        outputHistory = self.getPredicate(self._outputHistory, sessionID)
        try: that = outputHistory[-1]
        except IndexError: that = ""
        subbedThat = self._subbers['normal'].sub(that)
        response = ""	

	# fetch the current topic
	topic = unicode(self.getPredicate("topic", sessionID)).upper()
	subbedTopic = self._subbers['normal'].sub(topic)
	# Determine the final response.
	elem = self._brain.match(input, subbedThat, topic, False)
	self.element = elem
	#self.match_len = self._brain._returnMatchLen()
	#print str(elem)

	# Check element 
	if elem is None:
	    if self._verboseMode:
	        err = "WARNING: No match found for input: %s\n" % input.encode(self._textEncoding)
	        sys.stderr.write(err)
	else:
	    # Multistar '|' pattern match
	    if '***' in str(elem):
		response += elem.replace('***','')
	    else:
		# Try to process the element
		try:
			response += self._processElement(elem, sessionID).strip()
			response += " "
		except Exception as e:
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)
			print str(elem)
			pass

	    response = response.strip()

            # pop the top entry off the input stack.
            inputStack = self.getPredicate(self._inputStack, sessionID)
            inputStack.pop()
            self.setPredicate(self._inputStack, inputStack, sessionID)
        
	    if response.strip() == '.':
		return ""
	    else:
		return response.lower()


    def _processElement(self, elem, sessionID):
        """Process an AIML element.
        The first item of the elem list is the name of the element's
        XML tag.  The second item is a dictionary containing any
        attributes passed to that tag, and their values.  Any further
        items in the list are the elements enclosed by the current
        element's begin and end tags; they are handled by each
        element's handler function.
        """
        try:
            handlerFunc = self._elementProcessors[elem[0]]
        except:
            # Oops -- there's no handler function for this element
            # type!
            if self._verboseMode:
                err = "WARNING: No handler found for <%s> element\n" % elem[0].encode(
                    self._textEncoding, 'replace')
                sys.stderr.write(err)
            return ""
        return handlerFunc(elem, sessionID)

    ######################################################
    ### Individual element-processing functions follow ###
    ######################################################

    # <bot>
    def _processBot(self, elem, sessionID):
        """Process a <bot> AIML element.
        Required element attributes:
            name: The name of the bot predicate to retrieve.
        <bot> elements are used to fetch the value of global,
        read-only "bot predicates."  These predicates cannot be set
        from within AIML; you must use the setBotPredicate() function.
        """
        attrName = elem[1]['name']
        return self.getBotPredicate(attrName)

    # <condition>
    def _processCondition(self, elem, sessionID):
        """Process a <condition> AIML element.
        Optional element attributes:
            name: The name of a predicate to test.
            value: The value to test the predicate for.
        <condition> elements come in three flavors.  Each has different
        attributes, and each handles their contents differently.
        The simplest case is when the <condition> tag has both a 'name'
        and a 'value' attribute.  In this case, if the predicate
        'name' has the value 'value', then the contents of the element
        are processed and returned.
        If the <condition> element has only a 'name' attribute, then
        its contents are a series of <li> elements, each of which has
        a 'value' attribute.  The list is scanned from top to bottom
        until a match is found.  Optionally, the last <li> element can
        have no 'value' attribute, in which case it is processed and
        returned if no other match is found.
        If the <condition> element has neither a 'name' nor a 'value'
        attribute, then it behaves almost exactly like the previous
        case, except that each <li> subelement (except the optional
        last entry) must now include both 'name' and 'value'
        attributes.
        """
        attr = None
        response = ""
        attr = elem[1]

        # Case #1: test the value of a specific predicate for a
        # specific value.
        if 'name' in attr and 'value' in attr:
            val = self.getPredicate(attr['name'], sessionID)
            if val == attr['value']:
                for e in elem[2:]:
                    response += self._processElement(e, sessionID)
                return response
        else:
            # Case #2 and #3: Cycle through <li> contents, testing a
            # name and value pair for each one.
            try:
                name = None
                if 'name' in attr:
                    name = attr['name']
                # Get the list of <li> elemnents
                listitems = []
                for e in elem[2:]:
                    if e[0] == 'li':
                        listitems.append(e)
                # if listitems is empty, return the empty string
                if len(listitems) == 0:
                    return ""
                # iterate through the list looking for a condition that
                # matches.
                foundMatch = False
                for li in listitems:
                    try:
                        liAttr = li[1]
                        # if this is the last list item, it's allowed
                        # to have no attributes.  We just skip it for now.
                        if len(list(liAttr.keys())) == 0 and li == listitems[-1]:
                            continue
                        # get the name of the predicate to test
                        liName = name
                        if liName == None:
                            liName = liAttr['name']
                        # get the value to check against
                        liValue = liAttr['value']
                        # do the test
                        if self.getPredicate(liName, sessionID) == liValue:
                            foundMatch = True
                            response += self._processElement(li, sessionID)
                            break
                    except:
                        # No attributes, no name/value attributes, no
                        # such predicate/session, or processing error.
                        if self._verboseMode:
                            print_("Something amiss -- skipping listitem", li)
                        raise
                if not foundMatch:
                    # Check the last element of listitems.  If it has
                    # no 'name' or 'value' attribute, process it.
                    try:
                        li = listitems[-1]
                        liAttr = li[1]
                        if not ('name' in liAttr or 'value' in liAttr):
                            response += self._processElement(li, sessionID)
                    except:
                        # listitems was empty, no attributes, missing
                        # name/value attributes, or processing error.
                        if self._verboseMode:
                            print_("error in default listitem")
                        raise
            except:
                # Some other catastrophic cataclysm
                if self._verboseMode:
                    print_("catastrophic condition failure")
                raise
        return response

    # <date>
    def _processDate(self, elem, sessionID):
        """Process a <date> AIML element.
        <date> elements resolve to the current date and time.  The
        AIML specification doesn't require any particular format for
        this information, so I go with whatever's simplest.
        """
        return time.asctime()

    # <formal>
    def _processFormal(self, elem, sessionID):
        """Process a <formal> AIML element.
        <formal> elements process their contents recursively, and then
        capitalize the first letter of each word of the result.
        """
        response = ""
        for e in elem[2:]:
            response += self._processElement(e, sessionID)
        return string.capwords(response)

    # <gender>
    def _processGender(self, elem, sessionID):
        """Process a <gender> AIML element.
        <gender> elements process their contents, and then swap the
        gender of any third-person singular pronouns in the result.
        This subsitution is handled by the aiml.WordSub module.
        """
        response = ""
        for e in elem[2:]:
            response += self._processElement(e, sessionID)
        return self._subbers['gender'].sub(response)

    # <get>
    def _processGet(self, elem, sessionID):
        """Process a <get> AIML element.
        Required element attributes:
            name: The name of the predicate whose value should be
            retrieved from the specified session and returned.  If the
            predicate doesn't exist, the empty string is returned.
        <get> elements return the value of a predicate from the
        specified session.
        """
	test = self.getPredicate(elem[1]['name'], sessionID)
        if test: return test
	else: return "unknown"

    # <get>
    def _processCommand(self, elem, sessionID):
        """Process a <get> AIML element.
        Required element attributes:
            name: The name of the predicate whose value should be
            retrieved from the specified session and returned.  If the
            predicate doesn't exist, the empty string is returned.
        <get> elements return the value of a predicate from the
        specified session.
        """
	if 'memory.set' in elem[2].lower():
		mem = 'memory'+elem[2].split('memory')[1].split(')')[0]+')'
		try: exec(mem.replace('.Set','.set'))
		except Exception as e: print e
		return elem[2].replace(mem,'')
	else:
		return ''

    # <gossip>
    def _processGossip(self, elem, sessionID):
        """Process a <gossip> AIML element.
        <gossip> elements are used to capture and store user input in
        an implementation-defined manner, theoretically allowing the
        bot to learn from the people it chats with.  I haven't
        descided how to define my implementation, so right now
        <gossip> behaves identically to <think>.
        """
        return self._processThink(elem, sessionID)

    # <id>
    def _processId(self, elem, sessionID):
        """ Process an <id> AIML element.
        <id> elements return a unique "user id" for a specific
        conversation.  In PyAIML, the user id is the name of the
        current session.
        """
        return sessionID

    # <input>
    def _processInput(self, elem, sessionID):
        """Process an <input> AIML element.
        Optional attribute elements:
            index: The index of the element from the history list to
            return. 1 means the most recent item, 2 means the one
            before that, and so on.
        <input> elements return an entry from the input history for
        the current session.
        """
        inputHistory = self.getPredicate(self._inputHistory, sessionID)
        try:
            index = int(elem[1]['index'])
        except:
            index = 1
        try:
            return inputHistory[-index]
        except IndexError:
            if self._verboseMode:
                err = "No such index %d while processing <input> element.\n" % index
                sys.stderr.write(err)
            return ""

    # <javascript>
    def _processJavascript(self, elem, sessionID):
        """Process a <javascript> AIML element.
        <javascript> elements process their contents recursively, and
        then run the results through a server-side Javascript
        interpreter to compute the final response.  Implementations
        are not required to provide an actual Javascript interpreter,
        and right now PyAIML doesn't; <javascript> elements are behave
        exactly like <think> elements.
        """
        return self._processThink(elem, sessionID)

    # <learn>
    def _processLearn(self, elem, sessionID):
        """Process a <learn> AIML element.
        <learn> elements process their contents recursively, and then
        treat the result as an AIML file to open and learn.
        """
        filename = ""
        for e in elem[2:]:
            filename += self._processElement(e, sessionID)
        self.learn(filename)
        return ""

    # <li>
    def _processLi(self, elem, sessionID):
        """Process an <li> AIML element.
        Optional attribute elements:
            name: the name of a predicate to query.
            value: the value to check that predicate for.
        <li> elements process their contents recursively and return
        the results. They can only appear inside <condition> and
        <random> elements.  See _processCondition() and
        _processRandom() for details of their usage.
        """
        response = ""
        for e in elem[2:]:
            response += self._processElement(e, sessionID)
        return response

    # <lowercase>
    def _processLowercase(self, elem, sessionID):
        """Process a <lowercase> AIML element.
        <lowercase> elements process their contents recursively, and
        then convert the results to all-lowercase.
        """
        response = ""
        for e in elem[2:]:
            response += self._processElement(e, sessionID)
        return string.lower(response)

    # <person>
    def _processPerson(self, elem, sessionID):
        """Process a <person> AIML element.
        <person> elements process their contents recursively, and then
        convert all pronouns in the results from 1st person to 2nd
        person, and vice versa.  This subsitution is handled by the
        aiml.WordSub module.
        If the <person> tag is used atomically (e.g. <person/>), it is
        a shortcut for <person><star/></person>.
        """
        response = ""
        for e in elem[2:]:
            response += self._processElement(e, sessionID)
        if len(elem[2:]) == 0:  # atomic <person/> = <person><star/></person>
            response = self._processElement(['star', {}], sessionID)
        return self._subbers['person'].sub(response)

    # <person2>
    def _processPerson2(self, elem, sessionID):
        """Process a <person2> AIML element.
        <person2> elements process their contents recursively, and then
        convert all pronouns in the results from 1st person to 3rd
        person, and vice versa.  This subsitution is handled by the
        aiml.WordSub module.
        If the <person2> tag is used atomically (e.g. <person2/>), it is
        a shortcut for <person2><star/></person2>.
        """
        response = ""
        for e in elem[2:]:
            response += self._processElement(e, sessionID)
        if len(elem[2:]) == 0:  # atomic <person2/> = <person2><star/></person2>
            response = self._processElement(['star', {}], sessionID)
        return self._subbers['person2'].sub(response)

    # <random>
    def _processRandom(self, elem, sessionID):
        """Process a <random> AIML element.
        <random> elements contain zero or more <li> elements.  If
        none, the empty string is returned.  If one or more <li>
        elements are present, one of them is selected randomly to be
        processed recursively and have its results returned.  Only the
        chosen <li> element's contents are processed.  Any non-<li> contents are
        ignored.
        """
	listitems,listdesc = [],[]
	for e in elem[2:]:
	    if e[0] == 'li':
		listitems.append(e)
	if len(listitems) == 0:
	    return ""
	try:
		shuffle(listitems)
		return self._processElement(listitems[0], sessionID)
	except Exception as e:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		shuffle(listitems)
		return self._processElement(listitems[0], sessionID)

    # <sentence>
    def _processSentence(self, elem, sessionID):
        """Process a <sentence> AIML element.
        <sentence> elements process their contents recursively, and
        then capitalize the first letter of the results.
        """
        response = ""
        for e in elem[2:]:
            response += self._processElement(e, sessionID)
        try:
            response = response.strip()
            words = string.split(response, " ", 1)
            words[0] = string.capitalize(words[0])
            response = string.join(words)
            return response
        except IndexError:  # response was empty
            return ""

    # <set>
    def _processSet(self, elem, sessionID):
        """Process a <set> AIML element.
        Required element attributes:
            name: The name of the predicate to set.
        <set> elements process their contents recursively, and assign the results to a predicate
        (given by their 'name' attribute) in the current session.  The contents of the element
        are also returned.
        """
        value = ""
        for e in elem[2:]:
            value += self._processElement(e, sessionID)
        self.setPredicate(elem[1]['name'], value, sessionID)
        return value

    # <size>
    def _processSize(self, elem, sessionID):
        """Process a <size> AIML element.
        <size> elements return the number of AIML categories currently
        in the bot's brain.
        """
        return str(self.numCategories())

    # <sr>
    def _processSr(self, elem, sessionID):
        """Process an <sr> AIML element.
        <sr> elements are shortcuts for <srai><star/></srai>.
        """
        star = self._processElement(['star', {}], sessionID)
        response = self._respond(star, sessionID)
        return response

    # <srai>
    def _processSrai(self, elem, sessionID):
        """Process a <srai> AIML element.
        <srai> elements recursively process their contents, and then
        pass the results right back into the AIML interpreter as a new
        piece of input.  The results of this new input string are
        returned.
        """
        newInput = ""
        for e in elem[2:]:
            newInput += self._processElement(e, sessionID)
        return self._respond(newInput, sessionID)

    # <star>
    def _processStar(self, elem, sessionID):
        """Process a <star> AIML element.
        Optional attribute elements:
            index: Which "*" character in the current pattern should
            be matched?
        <star> elements return the text fragment matched by the "*"
        character in the current input pattern.  For example, if the
        input "Hello Tom Smith, how are you?" matched the pattern
        "HELLO * HOW ARE YOU", then a <star> element in the template
        would evaluate to "Tom Smith".
        """
        try:
            index = int(elem[1]['index'])
        except KeyError:
            index = 1
	# fetch the user's last input
	inputStack = self.getPredicate(self._inputStack, sessionID)
	input = self._subbers['normal'].sub(inputStack[-1])
	# fetch the Kernel's last response (for 'that' context)
	outputHistory = self.getPredicate(self._outputHistory, sessionID)
	try:
	    that = self._subbers['normal'].sub(outputHistory[-1])
	except:
	    that = ""  # there might not be any output yet
	topic = self.getPredicate('topic',sessionID)
	response = self._brain.star("star", input, that, topic, index)
	try:
		word = max(response.split(),key=len)
		tokens = nltk.word_tokenize(response)
		tagged = nltk.pos_tag(tokens)
		if self.predicateSet == False:
			test = t.filter(self._respond(word,sessionID))
			if test: response = re.split('[?.,!]',test)[0].lower().strip(string.punctuation)
	except: 
		if response != None: return response
		return None
        return response

    # <system>
    def _processSystem(self, elem, sessionID):
        """Process a <system> AIML element.
        <system> elements process their contents recursively, and then
        attempt to execute the results as a shell command on the
        server.  The AIML interpreter blocks until the command is
        complete, and then returns the command's output.
        For cross-platform compatibility, any file paths inside
        <system> tags should use Unix-style forward slashes ("/") as a
        directory separator.
        """
        # build up the command string
        command = ""
        for e in elem[2:]:
            command += self._processElement(e, sessionID)

        # normalize the path to the command.  Under Windows, this
        # switches forward-slashes to back-slashes; all system
        # elements should use unix-style paths for cross-platform
        # compatibility.
        #executable,args = command.split(" ", 1)
        #executable = os.path.normpath(executable)
        #command = executable + " " + args
        command = os.path.normpath(command)

        # execute the command.
        response = ""
        try:
            out = os.popen(command)
        except RuntimeError as msg:
            if self._verboseMode:
                err = "WARNING: RuntimeError while processing \"system\" element:\n%s\n" % msg.encode(
                    self._textEncoding, 'replace')
                sys.stderr.write(err)
            return "There was an error while computing my response.  Please inform my botmaster."
        time.sleep(0.01)  # I'm told this works around a potential IOError exception.
        for line in out:
            response += line + "\n"
        response = string.join(response.splitlines()).strip()
        return response

    # <template>
    def _processTemplate(self, elem, sessionID):
        """Process a <template> AIML element.
        <template> elements recursively process their contents, and
        return the results.  <template> is the root node of any AIML
        response tree.
        """
        response = ""
        for e in elem[2:]:
            response += self._processElement(e, sessionID)
        return response

    # text
    def _processText(self, elem, sessionID):
        """Process a raw text element.
        Raw text elements aren't really AIML tags. Text elements cannot contain
        other elements; instead, the third item of the 'elem' list is a text
        string, which is immediately returned. They have a single attribute,
        automatically inserted by the parser, which indicates whether whitespace
        in the text should be preserved or not.
        """
        try:
            elem[2] + ""
        except TypeError:
            raise TypeError("Text element contents are not text")

        # If the the whitespace behavior for this element is "default",
        # we reduce all stretches of >1 whitespace characters to a single
        # space.  To improve performance, we do this only once for each
        # text element encountered, and save the results for the future.
        if elem[1]["xml:space"] == "default":
            elem[2] = re.sub("\s+", " ", elem[2])
            elem[1]["xml:space"] = "preserve"
	return elem[2]

    # <that>
    def _processThat(self, elem, sessionID):
        """Process a <that> AIML element.
        Optional element attributes:
            index: Specifies which element from the output history to
            return.  1 is the most recent response, 2 is the next most
            recent, and so on.
        <that> elements (when they appear inside <template> elements)
        are the output equivilant of <input> elements; they return one
        of the Kernel's previous responses.
        """
        outputHistory = self.getPredicate(self._outputHistory, sessionID)
        index = 1
        try:
            # According to the AIML spec, the optional index attribute
            # can either have the form "x" or "x,y". x refers to how
            # far back in the output history to go.  y refers to which
            # sentence of the specified response to return.
            index = int(elem[1]['index'].split(',')[0])
        except:
            pass
        try:
            return outputHistory[-index]
        except IndexError:
            if self._verboseMode:
                err = "No such index %d while processing <that> element.\n" % index
                sys.stderr.write(err)
            return ""

    # <thatstar>
    def _processThatstar(self, elem, sessionID):
        """Process a <thatstar> AIML element.
        Optional element attributes:
            index: Specifies which "*" in the <that> pattern to match.
        <thatstar> elements are similar to <star> elements, except
        that where <star/> returns the portion of the input string
        matched by a "*" character in the pattern, <thatstar/> returns
        the portion of the previous input string that was matched by a
        "*" in the current category's <that> pattern.
        """
        try:
            index = int(elem[1]['index'])
        except KeyError:
            index = 1
        # fetch the user's last input
        inputStack = self.getPredicate(self._inputStack, sessionID)
        input = self._subbers['normal'].sub(inputStack[-1])
        # fetch the Kernel's last response (for 'that' context)
        outputHistory = self.getPredicate(self._outputHistory, sessionID)
        try:
            that = self._subbers['normal'].sub(outputHistory[-1])
        except:
            that = ""  # there might not be any output yet
        topic = self.getPredicate("topic", sessionID)
        response = self._brain.star("thatstar", input, that, topic, index)
        return response

    # <think>
    def _processThink(self, elem, sessionID):
        """Process a <think> AIML element.
        <think> elements process their contents recursively, and then
        discard the results and return the empty string.  They're
        useful for setting predicates and learning AIML files without
        generating any output.
        """
        for e in elem[2:]:
            self._processElement(e, sessionID)
        return ""

    # <topicstar>
    def _processTopicstar(self, elem, sessionID):
        """Process a <topicstar> AIML element.
        Optional element attributes:
            index: Specifies which "*" in the <topic> pattern to match.
        <topicstar> elements are similar to <star> elements, except
        that where <star/> returns the portion of the input string
        matched by a "*" character in the pattern, <topicstar/>
        returns the portion of current topic string that was matched
        by a "*" in the current category's <topic> pattern.
        """
        try:
            index = int(elem[1]['index'])
        except KeyError:
            index = 1
        # fetch the user's last input
        inputStack = self.getPredicate(self._inputStack, sessionID)
        input = self._subbers['normal'].sub(inputStack[-1])
        # fetch the Kernel's last response (for 'that' context)
        outputHistory = self.getPredicate(self._outputHistory, sessionID)
        try:
            that = self._subbers['normal'].sub(outputHistory[-1])
        except:
            that = ""  # there might not be any output yet
        topic = self.getPredicate("topic", sessionID)
        response = self._brain.star("topicstar", input, that, topic, index)
        return response

    # <uppercase>
    def _processUppercase(self, elem, sessionID):
        """Process an <uppercase> AIML element.
        <uppercase> elements process their contents recursively, and
        return the results with all lower-case characters converted to
        upper-case.
        """
        response = ""
        for e in elem[2:]:
            response += self._processElement(e, sessionID)
        return string.upper(response)

    # <version>
    def _processVersion(self, elem, sessionID):
        """Process a <version> AIML element.
        <version> elements return the version number of the AIML
        interpreter.
        """
        return self.version()


##################################################
### Self-test functions follow                 ###
##################################################
def _testTag(kern, tag, input, outputList):
    """Tests 'tag' by feeding the Kernel 'input'.  If the result
    matches any of the strings in 'outputList', the test passes.
    """
    global _numTests, _numPassed
    _numTests += 1
    print_("Testing <" + tag + ">:", end=' ')
    response = kern.respond(input).decode(kern._textEncoding)
    if response in outputList:
        print_("PASSED")
        _numPassed += 1
        return True
    else:
        print_("FAILED (response: '%s')" % response.encode(kern._textEncoding, 'replace'))
        return False


if __name__ == "__main__":
    # Run some self-tests
    k = Kernel()
    k.bootstrap(learnFiles="self-test.aiml")

    global _numTests, _numPassed
    _numTests = 0
    _numPassed = 0

    _testTag(k, 'bot', 'test bot', ["My name is Nameless"])

    k.setPredicate('gender', 'male')
    _testTag(k, 'condition test #1', 'test condition name value', ['You are handsome'])
    k.setPredicate('gender', 'female')
    _testTag(k, 'condition test #2', 'test condition name value', [''])
    _testTag(k, 'condition test #3', 'test condition name', ['You are beautiful'])
    k.setPredicate('gender', 'robot')
    _testTag(k, 'condition test #4', 'test condition name', ['You are genderless'])
    _testTag(k, 'condition test #5', 'test condition', ['You are genderless'])
    k.setPredicate('gender', 'male')
    _testTag(k, 'condition test #6', 'test condition', ['You are handsome'])

    # the date test will occasionally fail if the original and "test"
    # times cross a second boundary.  There's no good way to avoid
    # this problem and still do a meaningful test, so we simply
    # provide a friendly message to be print_ed if the test fails.
    date_warning = """
    NOTE: the <date> test will occasionally report failure even if it
    succeeds.  So long as the response looks like a date/time string,
    there's nothing to worry about.
    """
    if not _testTag(k, 'date', 'test date', ["The date is %s" % time.asctime()]):
        print_(date_warning)

    _testTag(k, 'formal', 'test formal', ["Formal Test Passed"])
    _testTag(k, 'gender', 'test gender', ["He'd told her he heard that her hernia is history"])
    _testTag(k, 'get/set', 'test get and set', ["I like cheese. My favorite food is cheese"])
    _testTag(k, 'gossip', 'test gossip', ["Gossip is not yet implemented"])
    _testTag(k, 'id', 'test id', ["Your id is _global"])
    _testTag(k, 'input', 'test input', ['You just said: test input'])
    _testTag(k, 'javascript', 'test javascript', ["Javascript is not yet implemented"])
    _testTag(k, 'lowercase', 'test lowercase', ["The Last Word Should Be lowercase"])
    _testTag(k, 'person', 'test person', ['HE think i knows that my actions threaten him and his.'])
    _testTag(k, 'person2', 'test person2', [
             'YOU think me know that my actions threaten you and yours.'])
    _testTag(k, 'person2 (no contents)', 'test person2 I Love Lucy', ['YOU Love Lucy'])
    _testTag(k, 'random', 'test random', ["response #1", "response #2", "response #3"])
    _testTag(k, 'random empty', 'test random empty', ["Nothing here!"])
    _testTag(k, 'sentence', "test sentence", ["My first letter should be capitalized."])
    _testTag(k, 'size', "test size", ["I've learned %d categories" % k.numCategories()])
    _testTag(k, 'sr', "test sr test srai", ["srai results: srai test passed"])
    _testTag(k, 'sr nested', "test nested sr test srai", ["srai results: srai test passed"])
    _testTag(k, 'srai', "test srai", ["srai test passed"])
    _testTag(k, 'srai infinite', "test srai infinite", [""])
    _testTag(k, 'star test #1', 'You should test star begin', ['Begin star matched: You should'])
    _testTag(k, 'star test #2', 'test star creamy goodness middle',
             ['Middle star matched: creamy goodness'])
    _testTag(k, 'star test #3', 'test star end the credits roll',
             ['End star matched: the credits roll'])
    _testTag(k, 'star test #4', 'test star having multiple stars in a pattern makes me extremely happy',
             ['Multiple stars matched: having, stars in a pattern, extremely happy'])
    _testTag(k, 'system', "test system", ["The system says hello!"])
    _testTag(k, 'that test #1', "test that", ["I just said: The system says hello!"])
    _testTag(k, 'that test #2', "test that", ["I have already answered this question"])
    _testTag(k, 'thatstar test #1', "test thatstar", ["I say beans"])
    _testTag(k, 'thatstar test #2', "test thatstar", ["I just said \"beans\""])
    _testTag(k, 'thatstar test #3', "test thatstar multiple",
             ['I say beans and franks for everybody'])
    _testTag(k, 'thatstar test #4', "test thatstar multiple", ['Yes, beans and franks for all!'])
    _testTag(k, 'think', "test think", [""])
    k.setPredicate("topic", "fruit")
    _testTag(k, 'topic', "test topic", ["We were discussing apples and oranges"])
    k.setPredicate("topic", "Soylent Green")
    _testTag(k, 'topicstar test #1', 'test topicstar', ["Solyent Green is made of people!"])
    k.setPredicate("topic", "Soylent Ham and Cheese")
    _testTag(k, 'topicstar test #2', 'test topicstar multiple', [
             "Both Soylents Ham and Cheese are made of people!"])
    _testTag(k, 'unicode support', "ÔÇÉÏºÃ", ["Hey, you speak Chinese! ÔÇÉÏºÃ"])
    _testTag(k, 'uppercase', 'test uppercase', ["The Last Word Should Be UPPERCASE"])
    _testTag(k, 'version', 'test version', ["PyAIML is version %s" % k.version()])
    _testTag(k, 'whitespace preservation', 'test whitespace', [
             "Extra   Spaces\n   Rule!   (but not in here!)    But   Here   They   Do!"])

    # Report test results
    print_("--------------------")
    if _numTests == _numPassed:
        print_("%d of %d tests passed!" % (_numPassed, _numTests))
    else:
        print_("%d of %d tests passed (see above for detailed errors)" % (_numPassed, _numTests))

    # Run an interactive interpreter
    print_("\nEntering interactive mode (ctrl-c to exit)")
    while True:
	print_(k.respond(input("> ")))
