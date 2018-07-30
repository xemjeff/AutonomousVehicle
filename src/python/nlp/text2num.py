#Test
import re
import string
from text2num4num import text2num4num

Small = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13,
    'fourteen': 14,
    'fifteen': 15,
    'sixteen': 16,
    'seventeen': 17,
    'eighteen': 18,
    'nineteen': 19,
    'twenty': 20,
    'thirty': 30,
    'forty': 40,
    'fifty': 50,
    'sixty': 60,
    'seventy': 70,
    'eighty': 80,
    'ninety': 90
}

Magnitude = {
    'thousand':     1000,
    'million':      1000000,
    'billion':      1000000000,
    'trillion':     1000000000000,
    'quadrillion':  1000000000000000,
    'quintillion':  1000000000000000000,
    'sextillion':   1000000000000000000000,
    'septillion':   1000000000000000000000000,
    'octillion':    1000000000000000000000000000,
    'nonillion':    1000000000000000000000000000000,
    'decillion':    1000000000000000000000000000000000,
    'tenth': 	    0.1,
    'hundredth':    0.01,
    'thousandth':   0.001,
    'millionth':    0.000001,
}

math = {
	"plus" : "+",
	"added" : "+",
	"minus" : "-",
	"subtract" : "-",
	"subtracted" : "-",
	"multiply" : "*",
	"multiplied" : "*",
	"times" : "*",
	"divide" : "/",
	"divided" : "/",
	"power" : "**",
	"modulo" : "%",
	"modulus" : "%",
} 

pattern = re.compile(r'\b(' + '|'.join(math.keys()) + r')\b')

class NumberException(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)
# Checks if there's a number in string
def hasNumber(s):
    return any(i.isdigit() for i in s)

# Find calculation
def text2num(s):
    try:
	    s = pattern.sub(lambda x: math[x.group()], s)
	    print "s: " + str(s)
	    a = re.split(r"[\s]+", s)
	    n = 0
	    g = 0
	    last_n = 0
	    last_g = 0
	    final = 0
	    last_w = ''
	    return_list = []
	    w_index = 0
	    for w in a:
		print "-------W: " + str(w)
		# Remove punctuation
		old_w = w
		if w.translate(None, string.punctuation) != '':
			w = w.translate(None, string.punctuation)
		# Math operator in w - skip
		if ((w in Magnitude.keys() == False) or ((w in math.values()) == True)) and (((w in Small.keys()) == False) or (text2num4num(w) == 'no')):
			print "G: " + str(g)
			if ((w in math.values()) == True) and (g == 0):
				return_list.append(str(w))
			elif ((w in math.values()) == True) and (g != 0):
				return_list.append(str(w))
				g = 0
			else:
				return_list.append(str(g))
				g = 0
			print "Skip '%s'\n" % w
		else:
			print "W: "+str(w)
			x = Small.get(w, None)
			if x is not None:
			    g += x
			elif w == "hundred" and g != 0:
			    g *= 100
			# Magnitudes processing (millionth, etc.)
			else:
			    x = Magnitude.get(w, None)
			    if x is not None:
				n += g * x
				g = 0
			print "G: " + str(g)
			print "N: " + str(n)
			if (g == 0) and (n != 0):
				return_list.append(str(n))
				n = 0
			elif ((g != 0) and (n == 0)):
				try:
					# If we got to a magnitude, then append the final figure
					if (w in Magnitude.keys() == True) and ((((len(a)-1) < (w_index+1)) and (a[w_index+1] != 'hundred')) and ((text2num4num(a[w_index+1]) == 'no') or ((a[w_index+1] in Small.keys()) == True))):
						return_list.append(str(n))
						n = 0
					# If we're at Small number in w, but next w isnt a number
					elif ((w in Small.keys()) == True) and ((((len(a)-1) < (w_index+1)) and (a[w_index+1] != 'hundred')) and ((text2num4num(a[w_index+1]) == 'no') or (a[w_index+1] in Magnitude.keys() == True))):
						return_list.append(str(g))
						g = 0
					elif w == 'hundred':
						return_list.append(str(g))
						g = 0
				except Exception as e:
					return_list.append(str(g))
					g = 0
			elif hasNumber(w):
				return_list.append(str(w))
		print str(return_list)+'\n'
		w_index += 1
	    #print return_list
	    
	    # Organize the list to include proper parentheses if not already 
	    for x in range(0,len(return_list)):
		if hasNumber(return_list[x]) and (final == 0):
			final = return_list[x]
		elif hasNumber(return_list[x]):
			if hasNumber(return_list[x-1]):
				final = eval(str(final)+'+'+str(return_list[x]))
			else:
				final = eval(str(final)+str(return_list[x-1])+str(return_list[x]))
	    
	    evaluated = []
	    print "evalutated: " + str(evaluated)
	    print "return list: " + str(return_list)
	    sep = re.split('[-+*/]', ' '.join(return_list))
	    for x in range (0, len(sep)):
		evaluated.append(eval(sep[x].strip().replace(' ','+')))

	    if [s for s in return_list if not s.isdigit()]:
		return str(eval([s for s in return_list if not s.isdigit()][0].join(map(str, evaluated))))
	    else:
		return str(''.join(map(str, evaluated)))
    except Exception as e:
	return ''
    
if __name__ == "__main__":
    assert 1 == text2num("one")
    assert 12 == text2num("twelve")
    assert 72 == text2num("seventy two")
    assert 300 == text2num("three hundred")
    assert 1200 == text2num("twelve hundred")
    assert 12304 == text2num("twelve thousand three hundred four")
    assert 6000000 == text2num("six million")
    assert 6400005 == text2num("six million four hundred thousand five")
    assert 123456789012 == text2num("one hundred twenty three billion four hundred fifty six million seven hundred eighty nine thousand twelve")
    assert 4000000000000000000000000000000000 == text2num("four decillion")

