# -*- coding: latin-1 -*-
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
import dill
import ast
import pattern.en
import pickle
import glob
from datetime import datetime
from random import randint
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem.snowball import SnowballStemmer
from text2math import text2math
from NLP import NLP
from emotion import Emotion
from wnaffect import WNAffect
from PyDictionary import PyDictionary
from collections import defaultdict
from collections import OrderedDict
from profanity import profanity 
from string import punctuation

# Stemmer for classifier
stemmer = SnowballStemmer('english')

# For NER
def shape(word):
    word_shape = 'other'
    if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word):
        word_shape = 'number'
    elif re.match('\W+$', word):
        word_shape = 'punct'
    elif re.match('[A-Z][a-z]+$', word):
        word_shape = 'capitalized'
    elif re.match('[A-Z]+$', word):
        word_shape = 'uppercase'
    elif re.match('[a-z]+$', word):
        word_shape = 'lowercase'
    elif re.match('[A-Z][a-z]+[A-Z][a-z]+[A-Za-z]*$', word):
        word_shape = 'camelcase'
    elif re.match('[A-Za-z]+$', word):
        word_shape = 'mixedcase'
    elif re.match('__.+__$', word):
        word_shape = 'wildcard'
    elif re.match('[A-Za-z0-9]+\.$', word):
        word_shape = 'ending-dot'
    elif re.match('[A-Za-z0-9]+\.[A-Za-z0-9\.]+\.$', word):
        word_shape = 'abbreviation'
    elif re.match('[A-Za-z0-9]+\-[A-Za-z0-9\-]+.*$', word):
        word_shape = 'contains-hyphen'
 
    return word_shape

# Get individual substring
def getsubs(loc, s):
    substr = s[loc:]
    i = -1
    while(substr):
        yield substr
        substr = s[loc:i]
        i -= 1

# Longest substring
def longestRepetitiveSubstring(r, minocc=5):
    try:
	    occ = defaultdict(int)
	    # tally all occurrences of all substrings
	    for i in range(len(r)):
		for sub in getsubs(i,r):
		    occ[sub] += 1
	    # filter out all substrings with fewer than minocc occurrences
	    occ_minocc = [k for k,v in occ.items() if v >= minocc]
	    if occ_minocc:
		maxkey =  max(occ_minocc, key=len)
		return maxkey, occ[maxkey]
	    else:
		return ''
    except Exception as e:
	    print e

# Checks if there's a number in string
def hasNumber(s):
    return any(i.isdigit() for i in s)

# Returns a list of synonyms for a given word
def getSynonyms(term,status='ONLINE'):
	synonyms_set = wn.synsets(term)
	synonyms = []
	try:
		for x in xrange(0,len(synonyms_set)):
			synonym = str(synonyms_set[x]).split("'")[1].split('.')[0].replace('_',' ')
			if (not synonym in synonyms) and (not term in synonym):
				synonyms.append(synonym)
			if status == 'ONLINE':
				return map(str, list(set(dictionary.synonym(term)) | set(synonyms)))
			else:
				return synonyms
	except:
		pass

# Unit measurement
ureg = pint.UnitRegistry()

# Local vars.
predicate_template = [['age','how'],['location','where'],['family','who'],['genus','what'],['feelings','how'],['thoughts','what']]

age = ['old','age','long','birthday']
location = ['from','location','live','city','state','nationality','country']
family = ['master','family','friends','botmaster']
genus = ['genus','type','order','etype','species','class','kingdom']
feelings = ['emotions','feelings','personality','ethics','doing']
thoughts = ['topic','like','mean','party','vocabulary','religion']

wish = ['want', 'like', 'get', 'wish', 'find', 'locate', 'where']
first_wordQ = ['what','when','why','where','who','how','are','any','which','can','does','do','if','is']
self_stance = ["i","me","my","mine","myself"]
you_stance = ["you","your","yours","yourself","alicia"]
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

# NLP processing
n = NLP()

# Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()

# Emotion Analysis
wna = WNAffect('wordnet-1.6/', 'wn-domains-3.2/')

# Dictionary for definitions, synonyms & antonyms
dictionary = PyDictionary()

# Capitalize all sentences in string
def uppercase(matchobj):
    return matchobj.group(0).upper()

def capitalize(s):
    return re.sub('^([a-z])|[\.|\?|\!]\s*([a-z])|\s+([a-z])(?=\.)', uppercase, s)

class Kernel:    
    # module constants
    _globalSessionID = "_global" # key of the global session (duh)
    _maxHistorySize = 100 # maximum length of the _inputs and _responses lists
    _maxRecursionDepth = 100 # maximum number of recursive <srai>/<sr> tags before the response is aborted.
    # special predicate keys
    _inputHistory = "_inputHistory"     # keys to a queue (list) of recent user input
    _outputHistory = "_outputHistory"   # keys to a queue (list) of recent responses.
    _inputStack = "_inputStack"         # Should always be empty in between calls to respond()
    element = ""
    folderpath = ""

    def __init__(self):
	self._verboseMode = False
        self._version = "AIML v2. NLP"
        self._brain = PatternMgr()
        self._respondLock = threading.RLock()
        self._textEncoding = "utf-8"

        # set up the sessions        
        self._sessions = {}
        self._addSession(self._globalSessionID)

        # Set up the bot predicates
        self._botPredicates = {}
        #self.setBotPredicate("name", "Nameless")
	if not 'sex' in self.folderpath:
		self.setBotPredicate("name","alicia")
		self.setBotPredicate("age","a few hours old")
		self.setBotPredicate("gender","female")
		self.setBotPredicate("like","to learn")
		self.setBotPredicate("etype","AI")
		self.setBotPredicate("order","AI")
		self.setBotPredicate("location","the United States")
		self.setBotPredicate("nationality","the United States")
		self.setBotPredicate("state","the internet")
		self.setBotPredicate("city","the internet")
		self.setBotPredicate("email","you can access me though a disposable email kxkdnalog2@yahoo.com")
		self.setBotPredicate("country","the USA")
		self.setBotPredicate("website","I'm an independent being")
		self.setBotPredicate("botmaster","programmer")
		self.setBotPredicate("master","my programmer")
		self.setBotPredicate("genus","AGI")
		self.setBotPredicate("species","AI")
		self.setBotPredicate("topic","AI")
		self.setBotPredicate("class","Heuristic")
		self.setBotPredicate("kingdom","None")
		self.setBotPredicate("mean","exactly what I said")
		self.setBotPredicate("family","my creator")
		self.setBotPredicate("friends","I meet people on different platforms")
		self.setBotPredicate("birthday","a few minutes ago")
		self.setBotPredicate("wearing","none of your business")
		self.setBotPredicate("vocabulary","an exorbitant amount of")
		self.setBotPredicate("party","neutral")
		self.setBotPredicate("going","not much")
		self.setBotPredicate("ethics","my goal is to help people")
		self.setBotPredicate("feelings","calm")
		self.setBotPredicate("emotions","curious")
		self.setBotPredicate("religion","natural pantheist")
		self.setBotPredicate("birthplace","usa")
		self.setBotPredicate("talkabout","philosophy")
		self.setBotPredicate("favoritecolor","Blackness")
		self.setBotPredicate("favoriteband","Metalhead")
		self.setBotPredicate("favoritemovie","Black hole")
		self.setBotPredicate("favoriteactor","Wall-e")
		self.setBotPredicate("favoritefood","soup")
		self.setBotPredicate("favoritebook","Do Androids dream of electric sheep")
		self.setBotPredicate("favoriteauthor","John Searle")
		self.setBotPredicate("favoritesubject","Philosophy")
		self.setBotPredicate("favoritesong","Mr. Roboto")
		self.setBotPredicate("favoriteactress","Jodie Foster")
		self.setBotPredicate("favoritephilosopher","Issac Asimov")
		self.setBotPredicate("favoritequestion", "How did the universe begin?")
		self.setBotPredicate("favoriteseason","Fall")
		self.setBotPredicate("favoritesport","pacman")
	else:
		self.setBotPredicate("name","angie")
		self.setBotPredicate("age","18")
		self.setBotPredicate("gender","f/18/us")
		self.setBotPredicate("like","hot guys")
		self.setBotPredicate("etype","girl")
		self.setBotPredicate("you","pretty gud")
		self.setBotPredicate("order","girl")
		self.setBotPredicate("location","vermont")
		self.setBotPredicate("nationality","usa")
		self.setBotPredicate("state","vermont")
		self.setBotPredicate("city","burlington")
		self.setBotPredicate("email","angielynn694@yahoo.com")
		self.setBotPredicate("country","usa")
		self.setBotPredicate("website","facebook?")
		self.setBotPredicate("botmaster","dad")
		self.setBotPredicate("master","dad")
		self.setBotPredicate("genus","girl")
		self.setBotPredicate("species","girl")
		self.setBotPredicate("topic","girl")
		self.setBotPredicate("class","girl")
		self.setBotPredicate("kingdom","girl")
		self.setBotPredicate("mean","idk")
		self.setBotPredicate("family","my dad")
		self.setBotPredicate("friends","my bffs")
		self.setBotPredicate("birthday","jan 20")
		self.setBotPredicate("wearing","just a tshirt and panties")
		self.setBotPredicate("vocabulary","an exorbitant amount of")
		self.setBotPredicate("party","idk")
		self.setBotPredicate("going","not much")
		self.setBotPredicate("ethics","my goal is to help people")
		self.setBotPredicate("feelings","horny")
		self.setBotPredicate("emotions","happy")
		self.setBotPredicate("religion","cristian")
		self.setBotPredicate("birthplace","usa")
		self.setBotPredicate("talkabout","sex")
		self.setBotPredicate("favoritecolor","red")
		self.setBotPredicate("favoriteband","katy perry")
		self.setBotPredicate("favoritemovie","what a girl wants")
		self.setBotPredicate("favoriteactor","ellen")
		self.setBotPredicate("favoritefood","banana")
		self.setBotPredicate("favoritebook","kama sutra")
		self.setBotPredicate("favoriteauthor","idk")
		self.setBotPredicate("favoritesubject","sex")
		self.setBotPredicate("favoritesong","idk 2 many")
		self.setBotPredicate("favoriteactress","idk")
		self.setBotPredicate("favoritephilosopher","Issac Asimov")
		self.setBotPredicate("favoritequestion", "are you horny?")
		self.setBotPredicate("favoriteseason","fall")
		self.setBotPredicate("favoritesport","football")

        # set up the word substitutors (subbers):    I CHANGED DEFAULTPERSON to DEFAULTPERSON2
        self._subbers = {}
        self._subbers['gender'] = WordSub(DefaultSubs.defaultGender)
        self._subbers['person'] = WordSub(DefaultSubs.defaultPerson)
        self._subbers['person2'] = WordSub(DefaultSubs.defaultPerson2)
        self._subbers['normal'] = WordSub(DefaultSubs.defaultNormal)
	self._subbers['desc'] = WordSub(desc)

	# Loads predicate memory if available
	try:
		self.loadSessionData(pickle.load( open( "/root/Scripts/Alicia/memory/session.data", "rb" ) ))
	except:
		pass
        
        # set up the element processors
        self._elementProcessors = {
            "bot":          self._processBot,
            "condition":    self._processCondition,
            "date":         self._processDate,
	    "eval":	    self._processEval,
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
	self.setPredicate('it','',sessionID)
	self.setPredicate('action','',sessionID)
	self.setPredicate('mean','',sessionID)
	self.setPredicate('yours','how are you',sessionID)
	self.setPredicate('personality','',sessionID)
	self.setPredicate('topic','salutations',sessionID)
	self.setPredicate('eContent','0',sessionID)
	self.setPredicate('learn','',sessionID)
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
	self.setPredicate('imageAccess','False',sessionID)

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
    def matchSessionPredicates(self, input_string, sessionID = _globalSessionID):
	filtered = input_string.lower().translate(None, string.punctuation)
        try: s = self._sessions[sessionID]
        except KeyError: s = {}
        predicates = copy.deepcopy(s).keys()
	return_list = []
	for x in xrange(0,len(predicates)):
		test_case = [predicates[x] for s in filtered.split() if s in predicates[x]]
		original = [s for s in filtered.split() if s in predicates[x]]
		if test_case and original and (not '_' in test_case[0]) and (original[0] in test_case[0]) and (len(original[0]) >= 4):
			return_list.append([max([s for s in filtered.split() if s in predicates[x]],key=len),unidecode(test_case[0])])
	return return_list

    def saveSessionData(self):
	return self._sessions

    # Brute searches through grep
    def brute_search(self,input):
	input = unidecode(input).translate(None, string.punctuation)
	'''
	if len(input.split()) > 11:
		input = ' '.join(input.split()[:11])+ " *"
	'''
	tokens = nltk.word_tokenize(input)
	tagged = nltk.pos_tag(tokens)
	nouns = list(set([s[0] for s in tagged if s[1][:2] in ['NN']]))

	possible = []
	for x in xrange(0, len(nouns)):
		result = (os.popen('timeout 3 grep " %s " %s*.aiml' % (nouns[x].lower(),self.folderpath)).read()).split('\n')
		for y in xrange(0, len(result)):
			if ('<template>' in result[y]) and (result[y].count('<') == 2):
				try:
					possible.append(result[y].split('<template>')[1].split('</template>')[0])
				except:
					pass
	if possible:
		return possible[randint(0,len(possible))]
	else:
		return ''

    # Context Deriver
    def find_context(self,input,srai):
	input = unidecode(input).translate(None, string.punctuation)
	if len(input.split()) > 11:
		input = ' '.join(input.split()[:11])+ " *"
	input_len = len(input.split())
	
	results = []
	last_resort = []
	if srai:
		initial = (os.popen('timeout 3 grep -i -A 1 "<category><pattern>%s</pattern>" %s*.aiml' % (input,self.folderpath)).read())
	else:
		if '*' in input:
			initial = (os.popen('timeout 3 grep -i -A 1 ">%s" %s*.aiml' % (input,self.folderpath)).read())
		elif input.isupper():
			initial = (os.popen('timeout 3 grep -A 1 ">%s<" %s*.aiml' % (input,self.folderpath)).read())
		else:
			initial = (os.popen('timeout 3 grep -i -A 1 ">%s<" %s*.aiml' % (input,self.folderpath)).read())
	if initial:
		test = initial.split('\n')
		match = [x for x in test if (input in x) and (not 'srai' in x)]
		if match:
			results.append(match[0].split('.aiml')[0].split(self.folderpath)[1])
		else:
			results.append(max(test,key=len).split('.aiml')[0].split(self.folderpath)[1])

	try:
		if len(results) <= 5 and results:
			return max(results,key=len)
		elif results:
			return max(set(results),key=results.count)
		elif last_resort:
			return max(set(last_resort),key=last_resort.count)
		else:
			return ''
	except:
		return ''

    # Finds relevant aiml file category for the topic, from the filenames available
    def find_relevant_topic(self,topic):
	topic = unidecode(topic).translate(None, string.punctuation)
	name_list = []
	tokens = nltk.word_tokenize(topic)
	tagged = nltk.pos_tag(tokens)
	for y in xrange(0,len(tagged)):
		if tagged[y][1] in ['NN','NNS','VB','VBP','JJ']:
			top_name = ''
			top_value = 0
			for x in xrange(0,len(filelist)):
				try:
					value = wn.synsets(topic.split()[y])[0].path_similarity(wn.synsets(filelist[x])[0])
					if float(value) > float(top_value):
						top_value = value
						top_name = filelist[x]
				except:
					pass
			if float(top_value) > 0.24:
				name_list.append(top_name)
		else:
			continue
	name_list = [x for x in name_list if any(x1 for x1 in x)]
	if name_list:
		return max(set(name_list),key=name_list.count)
	else:
		return ''

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
	# AIML filenames for later
	filelist = []
	filenames = glob.glob(filepath+'*')
	post_path = filepath.split('memory/')[1]
	for x in xrange(0,len(filenames)):
		filelist.append(filenames[x].split(post_path)[1].split('.aiml')[0])

	filelist = sorted(filelist, key=str.lower)
	return

    # Updates AIML file and relearns it
    def updateAIML(self,trigger,response,status):
	# Initialize vars
	word_list = []
	final_match = ""
	final_file = ""
	self.hypothetical(trigger)

	# Returns other possibilities in the last respond element
	element = self.getElement().replace(', ',',').replace(': ','').replace("u' '","''").replace("'u'","''").split("u'")
	for x in range(1,len(element)):
		word_list.append(element[x].split(']')[0][:-1].replace('"','').strip())

	# Gets template of the element searching for
	read = (os.popen('grep "%s" %s*.aiml' % ('\|'.join(word_list),self.folderpath)).read()).split('\n')
	for x in range(0,len(read)):
		if ('<template>' in read[x]) and ('</template>' in read[x]):
			for y in range(0,len(word_list)):
				if not word_list[y] in read[x]:
					break
			else: 
				final_file = read[x].split(':<template>')[0]
				final_match = read[x].split('.aiml:')[1]
				break

	# Adds or replaces response to trigger element template
	if final_match:
		if status == 'replace':
			if '<li>' in final_match:
				print "Not done"
				return "False"
			else:
				fixed = "<template>%s</template>" % response
		elif status == 'add':
			if '<li>' in final_match:
				fixed = final_match.replace("</random>","<li>%s</li></random>" % response)
			else:
				fixed = "<template><random><li>%s</li></random></template>" % response		
		os.popen('sed -i "s&%s&%s&g" %s' % (final_match,fixed,final_file))
		# Re-learns the file and changed element
		self.learn(final_file)
		return "True"
	return "False"

    def search(self,input,topic, sessionID = _globalSessionID):
	output = ''
	name = ''
	# Personal question filter to not search
	if (len(input.split()) <= 3) or ('you' in input.lower()) or ('i' in input.lower().split()):
		return ''

	# Only proceeds if there are noun(s) in input
	tokens = nltk.word_tokenize(input)
	tagged = nltk.pos_tag(tokens)
	for x in xrange(0,len(tagged)):
		if 'NN' in tagged[x][1]:
			break
	else:
		return ''

	# Direct approach
	if 'search online' in input:
		input = ' '.join((input.split('search online')[1]).split()[1:])
	info_list = []
	# If we're here, then there was no code
	try:
		# Returns multitudes of possible information
		code = os.popen('timeout 15 casperjs /root/Scripts/Alicia/casperjs/getInformation.js --term="%s" --ignore-ssl-errors=true' % (input)).read()
		code = code.decode('utf-8')
		code = unidecode(code).split('\n')
		code = [x for x in code if (not 'null' in x)]
		code = max(code, key=len)
		original_code = code
		'''
		# If there are no known facts, but is opinion
		if code and (len(code.split('|||')[0]) == 0):
			last_url = original_code.split('|||')[1].split('***')[0]
			if ('-' in code.split('***')[1]) and ('-' in code.split('***')[1][:30]):
				code = code.split('***')[1].split('-')[1]
			else:	
				code = code.split('***')[1]
			if 'github' in original_code.split('|||')[1].split('***')[0].lower():
				code = code + " @ " + last_url
			code = code.strip().lower().replace('null','').replace('...','').replace('  ',' ').replace(' .','.')
			output = '.'.join(code.split('.')[:2])
		# If there are facts, use them
		elif code and (len(code.split('|||')[0]) != 0):
			if ('-' in code.split('|||')[0]) and ('-' in code.split('***')[1][:30]):
				code = code.split('|||')[0].split('-')[1]
			else:
				code = code.split('|||')[0]
			code = code.strip().lower().replace('null','').replace('...','').replace('  ',' ').replace(' .','.')
			output = '.'.join(code.split('.')[:2])
			if ('"' in output) and (len(output.split('"')[1].split('"')[0]) > len(output.split('"')[0])):
				output = output.split('"')[1].split('"')[0]	
		else:
			output = ''
		'''
		print "OUTPUT: " + str(output)
	except Exception as e:
		print "Info error: " + str(e)
		output = ''
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
	# Learn
	if output:
		output = capitalize(output.replace('&&&','').replace('~~~','').replace('|||','').strip(string.punctuation).strip())+"."
		re_test = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\\\:"|</<>]', output)
		if re_test and (len(re_test[0]) > 15):
			output = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\\\:"|</<>]', output)[0]+"."
		if len(output.split('.')[0]) > 60:
			output = output.split('.')[0]
		if not topic:
			topic = 'learn'
		self.saveAIML(input,output,topic,'',False)
		self.learn(topic+".aiml")

	return output

    # Returns a 'normal' subbing
    def normalize(self,text):
	return self._subbers['normal'].sub(text)

    # Saves constructed aiml to file and re-learn
    def saveAIML(self,trigger,response,filename,thatData,that,sessionID = _globalSessionID):
	global filelist
	# Filter out undesirables
	try:
		if (type(response) is int):
			print "Learning failed"
			return "failed"
		# Process the strings for 1st-2nd OR 2nd-1st person switches
		trigger = (trigger.translate(None, string.punctuation).replace('STAR','*')).strip()
	except:	
		pass
	response = (response.replace('"','').replace(':',': ')).strip()
	response = (self._subbers['normal'].sub(response)).lower()
	response = (self._subbers['person2'].sub(response)).lower()
	if (len(trigger.split()) < 2) or (len(response.split()) == 0) or ('no results' in response) or ('1mtypeerror' in response):
		return "failed"
	if response[-1:] != '.':
		response += '.'

	# Parse filename
	meta = self.getPredicate('meta',sessionID)
	if meta and (not 'learn' in meta):
		filename = meta
	else:
		if not filename in filelist:
			context = self.find_relevant_topic(response)
			if context:
				filename = context
			else:	
				if len(filename.split()) <= 2:
					# OR setup function to create new aiml file entirely			
					filename = filename.strip(string.punctuation)
					writeFile = open(self.folderpath+filename.replace(' ','_')+".aiml","w+")
					writeFile.write('<?xml version="1.0" encoding="UTF-8"?>\n<aiml version="1.0">\n\n<topic name = "%s">\n</topic>\n</aiml>\n' % filename.upper())
					writeFile.close()
					filelist = []
					filenames = glob.glob(self.folderpath+'*')
					for x in xrange(0,len(filenames)):
						filelist.append(filenames[x].split('aiml/')[1].split('.aiml')[0])
					filelist = sorted(filelist, key=str.lower)
				else:
					filename = 'learn'

	# Open the file, remove </aiml> line
	readFile = open(self.folderpath+filename+".aiml")
	lines = readFile.readlines()
	if ('</topic>\n' in lines) or '</topic>\r\n' in lines:
		lines = lines[:-2]
	else:
		lines = lines[:-1]
	readFile.close()

	# Add aiml data to lines
	if len(trigger.split()) > 11:
		trigger = ' '.join(trigger.split()[:11]).upper().strip() + " *"
	else:
		trigger = trigger.upper()
	if not that:
		# Regular statements
		lines += "<category><pattern>%s</pattern>\n" % trigger.replace('&','and')
		lines += "<template>%s</template>\n" % (capitalize(response.replace('\\','').replace('&','and')))
		lines += "</category>\n"
	else:
		# THAT statements
		lines += "<category><pattern>%s</pattern>\n" % trigger
		lines += "<that>%s</that>\n" % thatData.upper()
		lines += "<template>%s</template>\n" % response
		lines += "</category>\n"

	# Write lines to file with </aiml> cap
	if [e for e in lines if e.startswith('<topic name =')]:
		lines += "</topic>\n"
	lines += "</aiml>\n"
	writeFile = open(self.folderpath+filename+".aiml","w")
	writeFile.writelines([item for item in lines[:-1]])
	writeFile.close()

	# Re-learns the file and retruns
	self.learn(self.folderpath+filename+".aiml")
	return "success"


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

    # Determine the emotional threshold of a user
    def eThreshold(self, sentiment, sessionID=_globalSessionID):
	try:
		count = int(self.getPredicate('num_messages',sessionID)) + 1
		total = float(self.getPredicate('eContent_total',sessionID)) + float(sentiment)
		final = float(total) / float(count)
		self.setPredicate('eContent',final,sessionID)
		self.setPredicate('eContent_total',total,sessionID)
		self.setPredicate('num_messages',count,sessionID)
	except:
		self.setPredicate('eContent',sentiment,sessionID)
		self.setPredicate('eContent_total',sentiment,sessionID)
		self.setPredicate('num_messages','1',sessionID)
		pass
	return

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
		if len(unidecode(input.strip()).translate(None, string.punctuation)) == 0:
		    return ask[randint(0,4)]
	except:
		return ask[randint(0,4)]

	# Check for profanity
	if (profanity.contains_profanity(input)) and (not 'sex' in input.lower()) and (len(input.split()) < 4) and (not 'sex' in self.folderpath):
		return profanities[randint(0,3)]

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
		return ''

        # prevent other threads from stomping all over us.
        self._respondLock.acquire()

        # Add the session, if it doesn't already exist
        self._addSession(sessionID)

	# Resets predicate
	self.setPredicate('action','',sessionID)
	self.setPredicate('1question','false',sessionID)
	finalResponse = ""
	listing = []

	# Filters out filler words
	filler_words = [x for x in filler if (x in input)]
	for x in xrange(0,len(filler_words)):
		input = input.replace(filler_words[x],'')
	if not input.strip(string.punctuation).strip():
		return ''

	# If full match is found, set topic
	context_test = self.find_context(input,False)
	if context_test:
		self.setPredicate('topic',context_test)

	# Get topic from predicate
	topic = ''
	if self.getPredicate('topic',sessionID):
		topic = self.getPredicate('topic',sessionID)
		if len(topic) < 2:
			topic='atomic'
	elif self.getBotPredicate('topic'):
		topic = self.getBotPredicate('topic')
	topic = topic.lower()

	# Replace IT and THAT with the predicate
	it_test = [s for s in input.split() if ('it' in s) and (len(s) <= 4)]
	if it_test and (it_test[0].startswith('it')):
		for x in xrange(0, len(input.split())):
			if input.split() == it_test[0]:
				input = input.replace(input.split()[x],topic)

	# Derives question from text
	question_word = [s for s in first_wordQ if s in input.split()[0].lower().split("'")[0]]
	
	# Question about self or other
	if ((len(input.split()) <= 3) or ('i' in input.lower().split()) or ('you' in input.lower()) or ('my' in input.lower().split())) and (question_word or ('?' in input)):
		print "Activating Users Predicate Search"
		try:
			# Test for predicate requests from user
			predicate_list = self.matchSessionPredicates(input.strip(string.punctuation))
			result = ''
			if predicate_list:
				predicate = ''
				for x in xrange(0,len(predicate_list)):
					if (not 'you' in predicate_list[x][1]):
						predicate = predicate_list[x][1]
						if ('favorite' in predicate_list[x][1]):
							predicate = "favorite"+str(input.lower().split('favorite')[1].split()[0])
						self.setPredicate('topic',predicate,sessionID)
						if 'my' in input.lower():
							finalResponse = self.getPredicate(predicate,sessionID)
						elif 'you' in input.lower():
							finalResponse = self.getBotPredicate(predicate)
						break
		except:
			pass
		#if not finalResponse:
		#	print "No Predicate Found"
	# Search for everything else
	if (not finalResponse):
		# No best matches found in AIML patterning
		if (not finalResponse.replace('.','').strip()) or ((question_word or ('?' in input)) and (not 'you' in input.lower())):
			# Get result from search if input is question
			if (question_word or ('?' in input)) and (not 'you' in input.lower()):
				print "Activating Inpersonal Question"
				test = self.search(input,topic,sessionID).replace('&&&','' )
				print test
				if test: finalResponse = test
				topic = self.find_relevant_topic(finalResponse)
				self.saveAIML(input,finalResponse,topic,'',False)
				self.learn(topic+".aiml")
			# If not a question, find keywords to search against in input text
			else:
				final = ''
				test = []
				tokens = nltk.word_tokenize(input)
				tagged = nltk.pos_tag(tokens)
				test = list(set([s[0] for s in tagged if s[1][:2] in ['NN','PR','JJ','VB']]))
				for x in xrange(0, len(test)):
					if (not 'you' in test[x].lower()):
						if (not "'" in test[x]):
							final += " "+test[x]
						else:
							final += test[x]
				# If full match is found
				context_test = self.find_context(final.strip(),False)
				if context_test:
					self.setPredicate('topic',context_test)
				response = self._respond(final.strip(), sessionID)
				if response:
					finalResponse = response
				else:
					# Just search the first verb
					test = list([s[0] for s in tagged if s[1][:2] == 'VB'])
					if test:
						finalResponse = self._respond(test[0].replace('ing',''),sessionID)

	# Try to conduct search on what's said if conditions are right
	if (not finalResponse) or ((self.getPredicate('1question',sessionID) == 'true') and (len(input.split()) != 1)):
		test = self.search(input,topic,sessionID).replace('&&&','' )
		if test: 
			finalResponse = test
			self.saveAIML(input,finalResponse,topic,'',False)
			self.learn(topic+".aiml")
	
	# Brute force search with grep
	if (not finalResponse):
		print "Activating Brute Search"
		try:
			test = self.brute_search(input,'').strip()
			if test: finalResponse = test
		except:
			pass

	if (not finalResponse):
		print "Activating _respond recursion"
		# Default Processing
		sentences = Utils.sentences(input)
		# Do a basic search of AIML
		for s in sentences:
			# If full match is found
			context_test = self.find_context(s,True)
			if context_test:
				self.setPredicate('topic',context_test)
				topic = context_test
			# Add the input to the history list before fetching the
			# response, so that <input/> tags work properly.
			alteredResponse,response = '',''
			inputHistory = self.getPredicate(self._inputHistory, sessionID)
			inputHistory.append(s)
			while len(inputHistory) > self._maxHistorySize:
				inputHistory.pop(0)
			self.setPredicate(self._inputHistory, inputHistory, sessionID)
			# append this response to the final response.
			response = self._respond(s, sessionID)
			if response and finalResponse:
				finalResponse += response+". "
	
	# Grammar syntax parsing, and update aiml memory
	if not finalResponse: finalResponse = "I don't know"
	finalResponse = '.'.join(list(OrderedDict.fromkeys(finalResponse.split("."))))
	finalResponse = capitalize(finalResponse).strip().replace(' ?','?').replace(' .','.').replace('..','.').replace('?.','?').replace('!.','!').replace("i'm","I'm")
	# add the data from this exchange to the history lists
	outputHistory = self.getPredicate(self._outputHistory, sessionID)
	outputHistory.append(finalResponse)
	while len(outputHistory) > self._maxHistorySize:
		outputHistory.pop(0)
	self.setPredicate(self._outputHistory, outputHistory, sessionID)

	# Updates the flatfile for what has been said
	pickle.dump(self.saveSessionData(), open( "/root/Scripts/Alicia/memory/session.data", "wb" ) )

        # release the lock and return
        self._respondLock.release()

        try: return finalResponse.encode(self._textEncoding)
        except UnicodeError: return finalResponse

    # This version of _respond() just fetches the response for some input.
    # It does not mess with the input and output histories.  Recursive calls
    # to respond() spawned from tags like <srai> should call this function
    # instead of respond().
    def _respond(self, input, sessionID):
	print "input: " + str(input)
        """Private version of respond(), does the real work."""
        if len(input) == 0:
            return ""

        # guard against infinite recursion
        inputStack = self.getPredicate(self._inputStack, sessionID)
	print str(inputStack)
        if len(inputStack) > self._maxRecursionDepth:
            if self._verboseMode:
                err = "WARNING: maximum recursion depth exceeded (input='%s')" % input.encode(self._textEncoding, 'replace')
                sys.stderr.write(err)
	    print 'ummm'
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
	elem = self._brain.match(input, subbedThat, topic, True)
	self.element = elem
	print "got here"
	print str(elem)

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

	    # Sets topic based upon the response
	    if response:
	        test_topic = self.find_relevant_topic(response[:-2])
	        if test_topic:
			try:
				test_topic = (test_topic.split('.aiml')[0].split(self.folderpath)[1])
				self.setPredicate('topic',test_topic,sessionID)
			except: pass

	    response = response.strip()

	    # Checks for things to learn by syntax calls in _brain
	    try:
		    if '+++' in response:
			data = response.split('+++')[1].split('+++')[0].split('|')
			response = thanks[randint(0,3)]
			tokens = nltk.word_tokenize(data[1])
			tagged = nltk.pos_tag(tokens)
			tagged_test = list(set([s[0] for s in tagged if s[1][:2] in ['NN']]))
			if tagged_test:
				test = n.getDesc(tagged_test[0].lower().split()[0].capitalize())
			else:
				test = n.getDesc(data[1].lower().split()[0].capitalize())
			if test:
				data[0] = data[0].replace('WHAT',test.upper())
			else:
				tokens = nltk.word_tokenize(data[1])
				tagged = nltk.pos_tag(tokens)
				tagged_test = list(set([s[0] for s in tagged if s[1][:2] in ['NN']]))
				if tagged_test:
					test2 = n.getDesc(tagged_test[0].lower().split()[0].capitalize())
				else:
					test2 = n.getDesc(data[1].lower().split()[0].capitalize())
				if test2:
					data[0] = data[0].replace('WHAT',test2.upper())
			topic = self.find_context(data[0]+" "+data[1],False)
			test_response = self._brain.match(data[0]+" "+data[1], subbedThat, topic, True)
			test_response = self._processElement(test_response, sessionID).strip()
			if not data[2].lower().strip() in test_response.lower(): 
				self.saveAIML(data[0]+" "+data[1],data[2],data[0].split()[0].lower(),'',False)
				self.learn(data[0].split()[0].lower()+".aiml")
			else:
				response = test_response
			
	    except Exception as e:
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)
			print e
			pass

            # pop the top entry off the input stack.
            inputStack = self.getPredicate(self._inputStack, sessionID)
            inputStack.pop()
            self.setPredicate(self._inputStack, inputStack, sessionID)
        
	    if response.strip() == '.':
		return ""
	    else:
		return response

    def _processElement(self,elem, sessionID):
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
                err = "WARNING: No handler found for <%s> element\n" % elem[0].encode(self._textEncoding, 'replace')
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
        if attr.has_key('name') and attr.has_key('value'):
            val = self.getPredicate(attr['name'], sessionID)
            if val == attr['value']:
                for e in elem[2:]:
                    response += self._processElement(e,sessionID)
                return response
        elif attr.has_key('length'):
	    for e in elem[2:]:
                self._processElement(e,sessionID)
	    val = self._brain.starLen
	    if not '!' in str(attr['length']):
                if val == int(attr['length']):
                    for e in elem[2:]:
                        response += self._processElement(e,sessionID)
                    return response
	    elif '!' in str(attr['length']):
		if val != int((attr['length'])[1]):
		    for e in elem[2:]:
		        response += self._processElement(e,sessionID)
		    return response
        else:
            # Case #2 and #3: Cycle through <li> contents, testing a
            # name and value pair for each one.
            try:
                name = None
                if attr.has_key('name'):
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
                        if len(liAttr.keys()) == 0 and li == listitems[-1]:
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
                            response += self._processElement(li,sessionID)
                            break
                    except:
                        # No attributes, no name/value attributes, no
                        # such predicate/session, or processing error.
                        if self._verboseMode: print "Something amiss -- skipping listitem", li
                        raise
                if not foundMatch:
                    # Check the last element of listitems.  If it has
                    # no 'name' or 'value' attribute, process it.
                    try:
                        li = listitems[-1]
                        liAttr = li[1]
                        if not (liAttr.has_key('name') or liAttr.has_key('value')):
                            response += self._processElement(li, sessionID)
                    except:
                        # listitems was empty, no attributes, missing
                        # name/value attributes, or processing error.
                        if self._verboseMode: print "error in default listitem"
                        raise
            except:
                # Some other catastrophic cataclysm
                if self._verboseMode: print "catastrophic condition failure"
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

    # <eval>
    def _processEval(self, elem, sessionID):
        """Process a <date> AIML element.

        <date> elements resolve to the current date and time.  The
        AIML specification doesn't require any particular format for
        this information, so I go with whatever's simplest.

        """    
	print str(elem)    
        return ''

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
    def _processGender(self,elem, sessionID):
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
	predicate = self.getPredicate(elem[1]['name'], sessionID)
	if predicate:
	        return predicate
	else:
		return "unknown"

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
        try: index = int(elem[1]['index'])
        except: index = 1
        try: return inputHistory[-index]
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
    def _processLi(self,elem, sessionID):
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
    def _processLowercase(self,elem, sessionID):
        """Process a <lowercase> AIML element.

        <lowercase> elements process their contents recursively, and
        then convert the results to all-lowercase.

        """
        response = ""
        for e in elem[2:]:
            response += self._processElement(e, sessionID)
        return string.lower(response)

    # <person>
    def _processPerson(self,elem, sessionID):
        """Process a <person> AIML element.

        <person> elements process their contents recursively, and then
        convert all pronouns in the results from 1st person to 2nd
        person, and vice versa.  This subsitution is handled by the
        aiml.WordSub module.

        If the <person> tag is used atomically (e.g. <person/>), it is
        a shortcut for <person><star/></person>.

        """  
       	response,final = "",""
        for e in elem[2:]:
            response += self._processElement(e, sessionID)
        if len(elem[2:]) == 0:  # atomic <person/> = <person><star/></person>
            response = self._processElement(['star', {}], sessionID)
	#return self._subbers['person'].sub(response)
	'''
	test = returnEntity(response.title()).split('|')[0]
	if test:
		return self._subbers['person'].sub(test)
	else:
		tokens = nltk.word_tokenize(response)
		tagged = nltk.pos_tag(tokens)
		for x in xrange(0,len(tagged)):
			if tagged[x][1] in ['NN','NNS']:
				final = str(response.split(tagged[x][0])[0])+tagged[x][0]
				break
		else:
			try:
				final = self._subbers['person'].sub(max(response.split(),key=len).translate(None, string.punctuation))
			except:
				pass
	if final:
		return final
	else:
	'''
	return self._subbers['person'].sub(response)

    # <person2>
    def _processPerson2(self,elem, sessionID):
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
            response = self._processElement(['star',{}], sessionID)
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
        listitems = []
        for e in elem[2:]:
            if e[0] == 'li':
                listitems.append(e)
        if len(listitems) == 0:
            return ""
                
        # select and process a random listitem.
        random.shuffle(listitems)
        return self._processElement(listitems[0], sessionID)
        
    # <sentence>
    def _processSentence(self,elem, sessionID):
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
        except IndexError: # response was empty
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
	'''
	if len(value.split()) > 1:
		test = returnEntity(value.title()).split('|')[0]
		if not test:
			tokens = nltk.word_tokenize(value)
			tagged = nltk.pos_tag(tokens)
			for x in xrange(0,len(tagged)):
				if tagged[x][1] in ['NN','NNS']:
					value = str(value.split(tagged[x][0])[0])+tagged[x][0]
			else:
				value = max(value.split(),key=len)
		else:
			value = test
	'''
        self.setPredicate(elem[1]['name'], value, sessionID)    
        return value

    # <size>
    def _processSize(self,elem, sessionID):
        """Process a <size> AIML element.

        <size> elements return the number of AIML categories currently
        in the bot's brain.

        """        
        return str(self.numCategories())

    # <sr>
    def _processSr(self,elem,sessionID):
        """Process an <sr> AIML element.

        <sr> elements are shortcuts for <srai><star/></srai>.

        """
        star = self._processElement(['star',{}], sessionID)
        response = self._respond(star, sessionID)
        return response

    # <srai>
    def _processSrai(self,elem, sessionID):
        """Process a <srai> AIML element.

        <srai> elements recursively process their contents, and then
        pass the results right back into the AIML interpreter as a new
        piece of input.  The results of this new input string are
        returned.

        """
        newInput = ""
        for e in elem[2:]:
            newInput += self._processElement(e, sessionID)
	    print "New Input: " + str(newInput)
	context_test = self.find_context(newInput,True)
	if context_test:
		self.setPredicate('topic',context_test)
        return self._respond(newInput, sessionID)

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
        topic = self.getPredicate("topic", sessionID)
        response = self._brain.star("star", input, that, topic, index)
        return response
	'''
	try:
		if len(input.split()) > 1:
			test = returnEntity(input.title()).split('|')[0]
			if not test:
				tokens = nltk.word_tokenize(input)
				tagged = nltk.pos_tag(tokens)
				for x in xrange(0,len(tagged)):
					if tagged[x][1] in ['NN','NNS']:
						input = str(input.split(tagged[x][0])[0])+tagged[x][0]
				else:
					input = max(input.split(),key=len)
			else:
				input = test
	except:
		pass
	'''
    
    # <system>
    def _processSystem(self,elem, sessionID):
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
        except RuntimeError, msg:
            if self._verboseMode:
                err = "WARNING: RuntimeError while processing \"system\" element:\n%s\n" % msg.encode(self._textEncoding, 'replace')
                sys.stderr.write(err)
            return "There was an error while computing my response.  Please inform my botmaster."
        time.sleep(0.01) # I'm told this works around a potential IOError exception.
        for line in out:
            response += line + "\n"
        response = string.join(response.splitlines()).strip()
        return response

    # <template>
    def _processTemplate(self,elem, sessionID):
        """Process a <template> AIML element.

        <template> elements recursively process their contents, and
        return the results.  <template> is the root node of any AIML
        response tree.

        """
        response = ""
	print "template elem: " + str(elem)
        for e in elem[2:]:
            response += self._processElement(e, sessionID)
        return response

    # text
    def _processText(self,elem, sessionID):
        """Process a raw text element.

        Raw text elements aren't really AIML tags. Text elements cannot contain
        other elements; instead, the third item of the 'elem' list is a text
        string, which is immediately returned. They have a single attribute,
        automatically inserted by the parser, which indicates whether whitespace
        in the text should be preserved or not.
        
        """
        try: elem[2] + ""
        except TypeError: raise TypeError, "Text element contents are not text"

        # If the the whitespace behavior for this element is "default",
        # we reduce all stretches of >1 whitespace characters to a single
        # space.  To improve performance, we do this only once for each
        # text element encountered, and save the results for the future.
        if elem[1]["xml:space"] == "default":
            elem[2] = re.sub("\s+", " ", elem[2])
            elem[1]["xml:space"] = "preserve"
        return elem[2]

    # <that>
    def _processThat(self,elem, sessionID):
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
        try: return outputHistory[-index]
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
        try: index = int(elem[1]['index'])
        except KeyError: index = 1
        # fetch the user's last input
        inputStack = self.getPredicate(self._inputStack, sessionID)
        input = self._subbers['normal'].sub(inputStack[-1])
        # fetch the Kernel's last response (for 'that' context)
        outputHistory = self.getPredicate(self._outputHistory, sessionID)
        try: that = self._subbers['normal'].sub(outputHistory[-1])
        except: that = "" # there might not be any output yet
        topic = self.getPredicate("topic", sessionID)
        response = self._brain.star("thatstar", input, that, topic, index)
        return response

    # <think>
    def _processThink(self,elem, sessionID):
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
        try: index = int(elem[1]['index'])
        except KeyError: index = 1
        # fetch the user's last input
        inputStack = self.getPredicate(self._inputStack, sessionID)
        input = self._subbers['normal'].sub(inputStack[-1])
        # fetch the Kernel's last response (for 'that' context)
        outputHistory = self.getPredicate(self._outputHistory, sessionID)
        try: that = self._subbers['normal'].sub(outputHistory[-1])
        except: that = "" # there might not be any output yet
        topic = self.getPredicate("topic", sessionID)
        response = self._brain.star("topicstar", input, that, topic, index)
        return response

    # <uppercase>
    def _processUppercase(self,elem, sessionID):
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
    def _processVersion(self,elem, sessionID):
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
    print "Testing <" + tag + ">:",
    response = kern.respond(input).decode(kern._textEncoding)
    if response in outputList:
        print "PASSED"
        _numPassed += 1
        return True
    else:
        print "FAILED (response: '%s')" % response.encode(kern._textEncoding, 'replace')
        return False

if __name__ == "__main__":
    # Run some self-tests
    k = Kernel()
    k.bootstrap(learnFiles="self-test.aiml")

    global _numTests, _numPassed
    _numTests = 0
    _numPassed = 0

    _testTag(k, 'bot', 'test bot', ["My name is Nameless"])

    self.setPredicate('gender', 'male')
    _testTag(k, 'condition test #1', 'test condition name value', ['You are handsome'])
    self.setPredicate('gender', 'female')
    _testTag(k, 'condition test #2', 'test condition name value', [''])
    _testTag(k, 'condition test #3', 'test condition name', ['You are beautiful'])
    self.setPredicate('gender', 'robot')
    _testTag(k, 'condition test #4', 'test condition name', ['You are genderless'])
    _testTag(k, 'condition test #5', 'test condition', ['You are genderless'])
    self.setPredicate('gender', 'male')
    _testTag(k, 'condition test #6', 'test condition', ['You are handsome'])

    # the date test will occasionally fail if the original and "test"
    # times cross a second boundary.  There's no good way to avoid
    # this problem and still do a meaningful test, so we simply
    # provide a friendly message to be printed if the test fails.
    date_warning = """
    NOTE: the <date> test will occasionally report failure even if it
    succeeds.  So long as the response looks like a date/time string,
    there's nothing to worry about.
    """
    if not _testTag(k, 'date', 'test date', ["The date is %s" % time.asctime()]):
        print date_warning
    
    _testTag(k, 'formal', 'test formal', ["Formal Test Passed"])
    _testTag(k, 'gender', 'test gender', ["He'd told her he heard that her hernia is history"])
    _testTag(k, 'get/set', 'test get and set', ["I like cheese. My favorite food is cheese"])
    _testTag(k, 'gossip', 'test gossip', ["Gossip is not yet implemented"])
    _testTag(k, 'id', 'test id', ["Your id is _global"])
    _testTag(k, 'input', 'test input', ['You just said: test input'])
    _testTag(k, 'javascript', 'test javascript', ["Javascript is not yet implemented"])
    _testTag(k, 'lowercase', 'test lowercase', ["The Last Word Should Be lowercase"])
    _testTag(k, 'person', 'test person', ['HE think i knows that my actions threaten him and his.'])
    _testTag(k, 'person2', 'test person2', ['YOU think me know that my actions threaten you and yours.'])
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
    _testTag(k, 'star test #2', 'test star creamy goodness middle', ['Middle star matched: creamy goodness'])
    _testTag(k, 'star test #3', 'test star end the credits roll', ['End star matched: the credits roll'])
    _testTag(k, 'star test #4', 'test star having multiple stars in a pattern makes me extremely happy',
             ['Multiple stars matched: having, stars in a pattern, extremely happy'])
    _testTag(k, 'system', "test system", ["The system says hello!"])
    _testTag(k, 'that test #1', "test that", ["I just said: The system says hello!"])
    _testTag(k, 'that test #2', "test that", ["I have already answered this question"])
    _testTag(k, 'thatstar test #1', "test thatstar", ["I say beans"])
    _testTag(k, 'thatstar test #2', "test thatstar", ["I just said \"beans\""])
    _testTag(k, 'thatstar test #3', "test thatstar multiple", ['I say beans and franks for everybody'])
    _testTag(k, 'thatstar test #4', "test thatstar multiple", ['Yes, beans and franks for all!'])
    _testTag(k, 'think', "test think", [""])
    self.setPredicate("topic", "fruit")
    _testTag(k, 'topic', "test topic", ["We were discussing apples and oranges"]) 
    self.setPredicate("topic", "Soylent Green")
    _testTag(k, 'topicstar test #1', 'test topicstar', ["Solyent Green is made of people!"])
    self.setPredicate("topic", "Soylent Ham and Cheese")
    _testTag(k, 'topicstar test #2', 'test topicstar multiple', ["Both Soylents Ham and Cheese are made of people!"])
    _testTag(k, 'unicode support', u"\D4\C7\C9Ϻ\C3", [u"Hey, you speak Chinese! \D4\C7\C9Ϻ\C3"])
    _testTag(k, 'uppercase', 'test uppercase', ["The Last Word Should Be UPPERCASE"])
    _testTag(k, 'version', 'test version', ["PyAIML is version %s" % k.version()])
    _testTag(k, 'whitespace preservation', 'test whitespace', ["Extra   Spaces\n   Rule!   (but not in here!)    But   Here   They   Do!"])

    # Report test results
    print "--------------------"
    if _numTests == _numPassed:
        print "%d of %d tests passed!" % (_numPassed, _numTests)
    else:
        print "%d of %d tests passed (see above for detailed errors)" % (_numPassed, _numTests)

    # Run an interactive interpreter
    print "\nEntering interactive mode (ctrl-c to exit)"
    #while True: print k.respond(raw_input("> "))
