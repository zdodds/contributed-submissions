#
# here is a string, named c
#

c = "claremont"

# let's see c[0], c[1], c[2], and c[3]
print("c[0] is", c[0])      # here, the index is 0
print("c[1] is", c[1])      # here, the index is 1
print("c[2] is", c[2])      # here, the index is 2
print("c[3] is", c[3])      # here, the index is 3


c = "claremont"

# let's see c[-1], c[-2], c[-3], and c[-4]
print("c[-1] is", c[-1])      # here, the index is 1
print("c[-2] is", c[-2])      # here, the index is 2
print("c[-3] is", c[-3])      # here, the index is 3
print("c[-4] is", c[-4])      # here, the index is 4


answer = c[0] + c[2] + c[3]
print("answer is", answer)


answer = c[0] + c[2] + c[-6]  # index 3 and index -6 are the same for 'claremont'
print("answer is", answer)


c[20]     # The index 20 is out of bounds... this will error


#
# 5 words from the 5Cs

# here are five 5C strings:

#    012345678     for reference
c = "claremont"
g = "graduate"
m = "mckenna"          # we'll skip m
s = "scripps"
p = "pitzer"
h = "harvey"
#    012345        for reference

# Your challenge: assemble one four-or-more letter word from each
# See if you can find a six letter word for one or more of them?!

# Example from c: real
print("Assembling real from c: ", c[3] + c[4] + c[2] + c[1])

# Example from c:    # find a different one from c:
print("Assembling clear from c: ", c[0] + c[2:5])
# Example from s:
print("Assembling price from s: ", s[-2] + s[2:4] + s[1] + p[-2])

# Example from s:
print("assembling crisp from s:", s[1:4] + s[-1] + s[-2])
# Example from p:
print("Assembling pitz from p:", p[0:4])
# Example from h:
print("Assembling very from h:", h[3:5] + h[2] + h[-1] )
print("Assembling heavy from h:", h[0] + h[-2] + h[1] + h[3] + h[-1])
p = "pitzer"
h = "harvey"
#    012345        for reference
print("Assembling tire from p: ", p[2:0:-1]  + p[-1:-3:-1] )


#    012345678     for reference
c = "claremont"
g = "graduate"
m = "mckenna"
s = "scripps"
p = "pitzer"
h = "harvey"
#    012345        for reference

# slicing examples

print("c[0:3] is ", c[0:3])
print("g[-3:] is ", g[-3:])
print("m[2:-2] is ", m[2:-2])
print("s[2:5] is ", s[2:5])
print("p[:2] is ", p[:2])      # what if an index is mising?
print("h[-2:] is ", h[-2:])    # ditto, for the other index


#
# 5 "cross-campus slices" from these 5C strings:
c = "claremont"
g = "graduate"
m = "mckenna"
s = "scripps"
p = "pitzer"
h = "harvey"

# Your challenge: assemble five four-or-more-letter words
# Each should use two-or-more-letters from two-or-more strings!
# It's ok to add single letters, in addition!
# positive or negative indices: up to you...  (your "slicing mood"?)

# Example 0 (ours):
print("Creating pithon: ", p[0:3] + h[0] + c[6:8])  # Python, almost!

# Example 1:
print("Creating pint: ", p[0:2] + c[7:]) #

# Example 2:
print("creating strict:", s[0] + p[2] + s[2:4] + c[0] + c[-1])
# Example 3:
print("creating lucky:", c[1] + g[4] + s[1] + m[2] + h[-1])
# Example 4:
print("creating practice:", p[0] + s[2:3] + g[2] + s[1] + c[-1] + p[1] + s[1] + m[3])
# Example 5:
print("creating ground:" , g[0:2] + c[-3] + g[-4] + m[-2] + g[3])


# 'harvey'  'claremont'


#
# This cell defines a function named   dbl

def dbl(x):
    """ Result: dbl returns twice its argument
        Argument x: a number (int or float)
    """
    return 2*x


# Try running this cell!

# dbl(23)    # this calls the function dbl


# Call dbl

dbl(23)


# call dbl with a string

dbl('wow ')


help(dbl)


def tpl(x):
    """ Return value: tpl returns thrice its argument
        Argument x: a number (int or float)
    """
    return 3*x


# Two tests are provided:
print("tpl(10) should be 30 <->", tpl(10))
print("tpl(46/3) should be 46 <->", tpl(46/3))


#
# space for your sq function  (use the previous examples to get started!)

def sq(x):
    """ Returns the square of x.
        Argument x: A number (int or float)
        Returns: The sq(x) of x (x * x)
    """
    return x * x

# Test cases
print("sq(10) should be 100 <->", sq(10))
print("sq(46) should be 2116 <->", sq(46))




# be sure these two tests work!
# you'll need to "comment them back in":

# print("sq(10) should be 100 <->", sq(10))
# print("sq(46) should be 2116 <->", sq(46))


#
# your checkends function





# be sure these four tests work!  Comment them in:

# print("checkends('no match') should be False <->", checkends('no match'))
# print("checkends('hah! a match') should be True <->", checkends('hah! a match'))

# print("checkends('q') should be True <->", checkends('q'))
# print("checkends(' ') should be True <->", checkends(' '))  # single space
def checkends(s):
    """Returns True if the first and last character of s are the same.

    Argument:
    s -- A string

    Returns:
    True if first and last character are the same, False otherwise.
    """
    return len(s) > 0 and s[0] == s[-1]

# Test cases
print("checkends('no match') should be False <->", checkends('no match'))
print("checkends('hah! a match') should be True <->", checkends('hah! a match'))
print("checkends('q') should be True <->", checkends('q'))
print("checkends(' ') should be True <->", checkends(' '))  # single space



# be sure to digest these examples of len

print("len('claremont') should be 9 <->", len('claremont'))  # 9 characters
print("len('this is a really long string') should be 28 <->", len('this is a really long string'))
print()

print("len('q') should be 1 <->", len('q'))    # one character, 'q'
print("len(' ') should be 1 <->", len(' '))    # single space, len is 1
print()

print("len('') should be 0 <->", len(''))    # single space, len is 0


#
# your flipside function

def flipside(s):
    """Returns a new string where the first half and second half of s are swapped.

    Argument: s -- A string

    Returns: A string with halves flipped.
    """
    x = len(s)//2
    return s[x:] + s[:x]


# be sure these tests work!  Comment them in:

print("flipside('homework') should be 'workhome' <->", flipside('homework'))
print("flipside('poptart') should be 'tartpop' <->", flipside('poptart'))    # shorter first half
print("flipside('carpet') should be 'petcar' <->", flipside('carpet'))
assert flipside('claremont')


#
# your transcribe_one function




# be sure these tests work!  Comment them in:

# print("transcribe_one('a') should be 'u' <->", transcribe_one('a'))
# print("transcribe_one('c') should be 'g' <->", transcribe_one('c'))
# print("transcribe_one('g') should be 'c' <->", transcribe_one('g'))
# print("transcribe_one('t') should be 'a' <->", transcribe_one('t'))
# print()

# print("transcribe_one('claremont') should be '' <->", transcribe_one('claremont'))  # will really look empty!
# print("transcribe_one('') should also be '' <->", transcribe_one(''))     # will also really look empty!
# print("transcribe_one('h') should also be '' <->", transcribe_one(''))    # will also really look empty!

def transcribe_one(s):
    """Transcribes a single DNA base (a, c, g, t) into its RNA complement.

    Argument:
    s -- A string of length 1 (single base)

    Returns:
    The RNA complement ('u' for 'a', 'g' for 'c', etc.) or '' for invalid input.
    """
    mapping = {'a': 'u', 'c': 'g', 'g': 'c', 't': 'a'}
    return mapping.get(s, '') if len(s) == 1 else ''

# Test cases
print("transcribe_one('a') should be 'u' <->", transcribe_one('a'))
print("transcribe_one('c') should be 'g' <->", transcribe_one('c'))
print("transcribe_one('g') should be 'c' <->", transcribe_one('g'))
print("transcribe_one('t') should be 'a' <->", transcribe_one('t'))
print("transcribe_one('claremont') should be '' <->", transcribe_one('claremont'))
print("transcribe_one('') should also be '' <->", transcribe_one(''))
print("transcribe_one('h') should also be '' <->", transcribe_one(''))    # will also really look empty!


#
# here, write convertFromSeconds(s)
#

def convertFromSeconds(s):
  """Converts seconds into [days, hours, minutes, seconds].

    Argument:
    s -- A nonnegative integer (total seconds)

    Returns:
    A list [days, hours, minutes, seconds].
  """
  days = s // (24 * 3600)
  s %= (24 * 3600)
  hours = s // 3600
  s %= 3600
  minutes = s // 60
  s %= 60
  seconds = s % 60
  return [days, hours, minutes, seconds]



# Try these tests!
print("convertFromSeconds(610) should be [0, 0, 10, 10] <->", convertFromSeconds(610))
print("convertFromSeconds(100000) should be [1, 3, 46, 40] <->", convertFromSeconds(100000))
print("convertFromSeconds(0) should be [0, 0, 0, 0] <->", convertFromSeconds(0))
print("convertFromSeconds(60) should be [0, 0, 1, 0] <->", convertFromSeconds(60))
print("convertFromSeconds(3600) should be [0, 1, 0, 0] <->", convertFromSeconds(3600))



#
# Example of a recursive function
#

def replace_i(s):
  """ input: a string s
      output: a new string like s, with
              all lowercase i's with !'s (exclamantion points)
  """
  if s == "":
    return s
  elif s[0] == "i":
    return "!" + replace_i(s[1:])
  else:
    return s[0] + replace_i(s[1:])

# tests!
print("replace_i('alien') should be      al!en      <->", replace_i('alien'))
print("replace_i('aiiiiiieee') should be a!!!!!!eee <->", replace_i('aiiiiiieee'))
print("replace_i('icily') should be      !c!ly      <->", replace_i('icily'))
print("replace_i('claremont') should be  claremont  <->", replace_i('claremont'))


#
# The vowel-counting example from class...
#

#
# vwl examples from class
#
def vwl(s):
    """vwl returns the number of vowels in s
       Argument: s, which will be a string
    """
    if s == '':
        return 0   # no vowels in the empty string
    elif s[0] in 'aeiou':
        return 1 + vwl(s[1:])   # count 1 for the vowel
    else:
        return 0 + vwl(s[1:])   # The 0 + isn't necessary but looks nice

#
# Tests!
print( "vwl('sequoia') should be 5 <-> ", vwl('sequoia') )
print( "vwl('bcdfg') should be 0 <-> ", vwl('bcdfg') )
print( "vwl('e') should be 1 <-> ", vwl('e') )
print( "vwl('This sentence has nine vowels.') should be 9 <-> ", vwl('This sentence has nine vowels.') )

#
# here are keepvwl and dropvwl, also from class:


#
# keepvwl example from class
#
def keepvwl(s):
    """keepvwl returns the vowels in s
       Argument: s, which will be a string
       Return value: a string with all of s's vowels, in order
    """
    if s == '':
        return ''   # return the empty string
    elif s[0] in 'aeiou':
        return s[0] + keepvwl(s[1:]) # keep s[0], since it's a vowel
    else:
        return '' + keepvwl(s[1:])   # '' isn't necessary but fits conceptually!


#
# dropvwl example from class
#
def dropvwl(s):
    """dropvwl returns the non-vowels in s.  Note that "non-vowels" includes
         characters that are not alphabetic.
       Argument: s, which will be a string
       Return value: s with the vowels removed
    """
    if s == '':
        return ''   # return the empty string
    elif s[0] in 'aeiou':
        return '' + dropvwl(s[1:])   # drop s[0], since it's a vowel
    else:
        return s[0] + dropvwl(s[1:])   # keep s[0], since it's NOT a vowel!




# fun - functions continue to exist in other cells
# try it:

replace_i('aliiien')   # a three-i'ed alien: exciting!!!


#
# A full transcribe function




# If you work on this, please create 4-6 tests showing how it does:

# tests here...
# model them from the above tests...



#
# here, write your letterScore function
#
def letterScore(c):
    """Returns Scrabble score for a single letter.

    Argument:
    c -- A single letter (uppercase or lowercase)

    Returns:
    The Scrabble score of the letter or 0 for non-letters.
    """
    score_dict = {
        1: "aeioulnstrAEIOULNSTR",
        2: "dgDG",
        3: "bcmpBCMP",
        4: "fhvwyFHVWY",
        5: "kK",
        8: "jxJX",
        10: "qzQZ"
    }
    for score, letters in score_dict.items():
        if c in letters:
            return score
    return 0

# Test cases
print("letterScore('h') should be 4 <->", letterScore('h'))
print("letterScore('c') should be 3 <->", letterScore('c'))
print("letterScore('a') should be 1 <->", letterScore('a'))
print("letterScore('z') should be 10 <->", letterScore('z'))
print("letterScore('^') should be 0 <->", letterScore('^'))



# be sure to try these tests!

# print( "letterScore('h') should be  4 :",  letterScore('h') )
# print( "letterScore('c') should be  3 :",  letterScore('c') )
# print( "letterScore('a') should be  1 :",  letterScore('a') )
# print( "letterScore('z') should be 10 :",  letterScore('z') )
# print( "letterScore('^') should be  0 :",  letterScore('^') )


#
# here, write your scabbleScore function
#

def scrabbleScore(S):
    """Returns the total Scrabble score for a string.

    Argument:
    S -- A string containing letters

    Returns:
    The total Scrabble score of the string.
    """
    if S == "":
        return 0
    else:
        return letterScore(S[0]) + scrabbleScore(S[1:])




# be sure to try these tests!

#
# Tests
#
print( "scrabbleScore('quetzal')           should be  25 :",  scrabbleScore('quetzal') )
print( "scrabbleScore('jonquil')           should be  23 :",  scrabbleScore('jonquil') )
print( "scrabbleScore('syzygy')            should be  25 :",  scrabbleScore('syzygy') )
print( "scrabbleScore('?!@#$%^&*()')       should be  0 :",  scrabbleScore('?!@#$%^&*()') )
print( "scrabbleScore('')                  should be  0 :",  scrabbleScore('') )
print( "scrabbleScore('abcdefghijklmnopqrstuvwxyz') should be  87 :",  scrabbleScore('abcdefghijklmnopqrstuvwxyz') )


#
# here, write your own string-transforming function
#       be sure to use recursion and write at least three tests...

def transform(s):
  """ include a docstring! """
  return s
  # this is not complete!!

def transform(s):
    """Returns a new string where all vowels in s are doubled.

    Argument:
    s -- A string

    Returns:
    A modified string where each vowel is repeated twice.
     """
    if s == "":
        return ""  # Base case: empty string returns empty
    elif s[0] in "aeiouAEIOU":
        return s[0] * 2 + double_vowels(s[1:])  # Double vowels
    else:
        return s[0] + double_vowels(s[1:])  # Keep consonants unchanged



# your tests here:
print("double_vowels('hello') should be 'heelloo' <->", double_vowels('hello'))
print("double_vowels('recursive') should be 'reecuursiivee' <->", double_vowels('recursive'))
print("double_vowels('AEIOU') should be 'AAEEIIOOUU' <->", double_vowels('AEIOU'))
print("double_vowels('Python') should be 'Pythoon' <->", double_vowels('Python'))
print("double_vowels('xyz') should be 'xyz' <->", double_vowels('xyz'))  # No vowels to change


#
# Here is space to write pigletLatin and pigLatin!
#



# be sure to try these tests!
# print( "pigLatin('string')             should be  'ingstray'   :",  pigLatin('string') )
# print( "pigLatin('yttrium')            should be  'yttriumway' :",  pigLatin('yttrium') )
# print( "pigLatin('yoohoo')             should be  'oohooyay'   :",  pigLatin('yoohoo') )
# print( "pigLatin('stymie')             should be  'ymiestay'   :",  pigLatin('stymie') )


