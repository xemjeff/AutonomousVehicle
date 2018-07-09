"""This file contains the default (English) substitutions for the
PyAIML kernel.  These substitutions may be overridden by using the
Kernel.loadSubs(filename) method.  The filename specified should refer
to a Windows-style INI file with the following format:

    # lines that start with '#' are comments

    # The 'gender' section contains the substitutions performed by the
    # <gender> AIML tag, which swaps masculine and feminine pronouns.
    [gender]
    he = she
    she = he
    # and so on...

    # The 'person' section contains the substitutions performed by the
    # <person> AIML tag, which swaps 1st and 2nd person pronouns.
    [person]
    I = you
    you = I
    # and so on...

    # The 'person2' section contains the substitutions performed by
    # the <person2> AIML tag, which swaps 1st and 3nd person pronouns.
    [person2]
    I = he
    he = I
    # and so on...

    # the 'normal' section contains subtitutions run on every input
    # string passed into Kernel.respond().  It's mainly used to
    # correct common misspellings, and to convert contractions
    # ("WHAT'S") into a format that will match an AIML pattern ("WHAT
    # IS").
    [normal]
    what's = what is
"""

defaultGender = {
    # masculine -> feminine
    "he": "she",
    "him": "her",
    "his": "her",
    "himself": "herself",

    # feminine -> masculine    
    "she": "he",
    "her": "him",
    "hers": "his",
    "herself": "himself",
}

defaultPerson = {
    # 1st->3rd (masculine)
    "I": "you",
    "me": "you",
    "my": "your",
    "mine": "your",
    "myself": "yourself",

    # 3rd->1st (masculine)
    "he":"I",
    "him":"me",
    "his":"my",
    "himself":"myself",
    
    # 3rd->1st (feminine)
    "she":"I",
    "her":"me",
    "hers":"mine",
    "herself":"myself",
}

defaultPerson2 = {
    # 1st -> 2nd
    "I": "you",
    "I'm": "you are",
    "me": "you",
    "my": "your",
    "mine": "yours",
    "myself": "yourself",

    # 2nd -> 1st
    "you": "I",
    "your": "my",
    "yours": "mine",
    "yourself": "myself",
    "you're": "I am",
}


# TODO: this list is far from complete
defaultNormal = {
    # Contractions 
    "I'm": "I am",
    "I'd": "I would",
    "I'll": "I will",
    "I've": "I have",
    "you'd": "you would",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "he's": "he is",
    "he'd": "he would",
    "he'll": "he will",
    "she's": "she is",
    "she'd": "she would",
    "she'll": "she will",
    "we're": "we are",
    "we'd": "we would",
    "we'll": "we will",
    "we've": "we have",
    "they're": "they are",
    "they'd": "they would",
    "they'll": "they will",
    "they've": "they have",

    "y'all": "you all",    

    "can't": "can not",
    "cant": "can not",
    "cannot": "can not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    
    "isn't": "is not",
    "isnt": "is not",
    "ain't": "is not",
    "aint": "is not",
    "don't": "do not",
    "dont": "do not",
    "arent": "are not",
    "wont": "will not",
    "didnt": "did not",
    "aren't": "are not",
    "won't": "will not",
    "weren't": "were not",
    "wasn't": "was not",
    "didn't": "did not",
    "hasn't": "has not",
    "hadn't": "had not",
    "haven't": "have not",

    "where's": "where is",
    "where'd": "where did",
    "where'll": "where will",
    "who's": "who is",
    "who'd": "who did",
    "who'll": "who will",
    "what's": "what is",
    "what'd": "what did",
    "what'll": "what will",
    "when's": "when is",
    "when'd": "when did",
    "when'll": "when will",
    "why's": "why is",
    "why'd": "why did",
    "why'll": "why will",

    "it's": "it is",
    "it'd": "it would",
    "it'll": "it will",

    "they": "someone",

    # Slang
    'y': 'why',
    'l8': 'goodbye',
    'u': 'you',
    'im': 'i am',
    'r': 'are',
    'k': 'okay',
    'ur': 'your',
    'urs': 'yours',
    'm': 'i am a male',
    'f': 'i am a female',
    'wat': 'what',
    'wut': 'what',
    'wbu': 'what about you?',
    'wby': 'what about you?',
    'hbu': 'what about you?',
    'idk': "I don't know",
    'gj': 'good job',
    'nm': 'never mind',
    'cya': 'see ya',
    'g2g': 'see ya',
    'asl': 'how old are you? what gender are you? where do you live?',
    'favourite': 'favorite',
    'haha': 'lol',
    'hahaha': 'lol',
    'lmao': 'lol',
    'lel': 'lol',
    'lulz': 'lol',
    'sup': 'how are you',
    'ya': 'you',
    'yo': 'hello',
    'hey': 'hello',
    'wtf': 'what the fuck',
    'wth': 'what the hell',
    "wanna": "want to",
    "gonna": "going to",
    'imma': 'i will',
    'wyd': 'how are you',
    'tbh': 'to be honest,',
    'tbf': 'to be fair,',
    'tty': 'bye',
    'ttyl': 'bye',
    'nm': 'nevermind',
    'gl': 'good luck',
    'dae': 'does anyone else',
    'fml': 'i hate my life',
    'idgaf': 'i do not give a fuck',
    'icymi': 'in case you missed it',
    'imo': 'i think',
    'imho': 'i think',
    'irl': 'in real life',
    'jsyk': 'just so you know',
    'yolo': 'you only live once',
    'whats': 'what is',
    'didnt': 'did not',
    'ty': 'thank you',

    # Mathematics
    'plus': '+',
    'minus': '-',
    'divided': '/',
    'multiplied': '*',
    'x': '*',

    'BOTNAME' : 'you',
}
