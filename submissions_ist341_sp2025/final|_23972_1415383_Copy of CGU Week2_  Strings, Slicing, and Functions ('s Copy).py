41+1


perc = 0.91
if perc > 0.95:
  print ('A')
elif perc > 0.90:
    print ('A-')
elif perc > 0.70:
      print ('Pass')
else:
        print ('Aargh')


perc = 0.80
if perc > 0.00:
  print ('Aargh!')
elif perc > 0.70:
  print ('Pass')
elif perc > 0.90:
  print ('A-')
else:
  print ('A')
 #I can't get an A- in this code cell!


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

# Example from c:   # find a different one from c:
print ("Assembling grade from g:", g[0]+g[1]+g[2]+g[3]+g[-1]) #Surely it's an A!

# Example from s:
print("Assembling crisps from s: ",s[1]+s[2]+s[3]+s[0]+s[-2]+s[-1])

# Example from p:
print("Assembling prizez from p:", p[0]+p[-1]+p[1]+p[3]+p[-2]+p[3])

# Example from h:
print("Assembling rave:", h[2]+h[1]+h[3]+h[-2])

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
print("Creating dna: ", g[3] + m[5:7])

# Example 3:
print("Creating the word 'crackers':", c[0] + g[1:3] + m[1:3] + p[4:6] + s[0])

# Example 4:
print("Creating the word 'scare':", s[0:2] + c[2:5])
# Example 5:
print("creating the word 'carve':", c[0] + c[2:4] + h[3:5])


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

def flipside(s):
    """flip the two sides."""
    x = len(s)//2
    return s[x:] + s[:x]

print(flipside('homework') + "should be 'workhome' <-> " + flipside('homework'))
print(flipside('poptart') + "should be 'tartpop' <-> " + flipside('poptart'))    # shorter first half
print(flipside('carpet') + "should be 'petcar' <-> " + flipside('carpet'))

# be sure these two tests work!
# you'll need to "comment them back in":
def sq(x):
  return x*x
print("sq(10) should be 100 <->", sq(10))
print("sq(46) should be 2116 <->", sq(46))


def checkends(s):
    """Check if the first and last characters of the string are the same."""
    if len(s) == 0:  # Handle empty string case
        return False
    return s[0] == s[-1]

# Test the function
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
    """flip the two sides."""
    x = len(s)//2
    return s[x:] + s[:x]


# be sure these tests work!  Comment them in:

print("flipside('homework') should be 'workhome' <->", flipside('homework'))
print("flipside('poptart') should be 'tartpop' <->", flipside('poptart'))    # shorter first half
print("flipside('carpet') should be 'petcar' <->", flipside('carpet'))


#
# your transcribe_one function

def transcribe_one(s):
  """This is a transcribe function"""
  if len(s) == 1:
    if s == 'a':
      return 'u'
    elif s == 'c':
      return 'g'
    elif s == 'g':
      return 'c'
    elif s == 't':
      return 'a'
  return ''


# be sure these tests work!  Comment them in:

print("transcribe_one('a') should be 'u' <->", transcribe_one('a'))
print("transcribe_one('c') should be 'g' <->", transcribe_one('c'))
print("transcribe_one('g') should be 'c' <->", transcribe_one('g'))
print("transcribe_one('t') should be 'a' <->", transcribe_one('t'))
print()

print("transcribe_one('claremont') should be '' <->", transcribe_one('claremont'))  # will really look empty!
print("transcribe_one('') should also be '' <->", transcribe_one(''))     # will also really look empty!
print("transcribe_one('h') should also be '' <->", transcribe_one(''))    # will also really look empty!



def convertFromSeconds(s):
    """
    Converts a number of seconds to days, hours, minutes, and seconds.

    Args:
        s: A non-negative integer representing the number of seconds.

    Returns:
        A list of four non-negative integers representing days, hours, minutes, and seconds.
    """
    days = s // (24 * 3600)  # Calculate days
    s %= (24 * 3600)         # Remaining seconds after extracting days
    hours = s // 3600        # Calculate hours
    s %= 3600               # Remaining seconds after extracting hours
    minutes = s // 60        # Calculate minutes
    s %= 60                 # Remaining seconds after extracting minutes
    seconds = s              # Remaining seconds

    return [days, hours, minutes, seconds]

# Test the function
print("convertFromSeconds(610) should return [0, 0, 10, 10] <->", convertFromSeconds(610))
print("convertFromSeconds(100000) should return [1, 3, 46, 40] <->", convertFromSeconds(100000))


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
def replace_i(s):
  """ input: a string s
      output: a new string like s, with
              all lowercase i's with !'s (exclamantion points)
  """
  return s.replace('i', '!')

replace_i('aliiien')   # a three-i'ed alien: exciting!!!


#
# A full transcribe function
def transcribe_one(s):
    if s == 'a':
        return 'u'
    elif s == 'c':
        return 'g'
    elif s == 'g':
        return 'c'
    elif s == 't':
        return 'a'
    else:
        return None  # Non-transcribable character
def transcribe(s):
  """This is a transcribe function"""
  if len(s) == 1:
    if s == 'a':
      return 'u'
    elif s == 'c':
      return 'g'
    elif s == 'g':
      return 'c'
    elif s == 't':
      return 'a'
  return ''



# If you work on this, please create 4-6 tests showing how it does:

# tests here...
print("transcribe('actg') should be 'ugac' <->", transcribe('actg'))
print("transcribe('aaa') should be 'uuu' <->", transcribe('aaa'))
print("transcribe('cgta') should be 'gcau' <->", transcribe('cgta'))
print("transcribe('') should be '' <->", transcribe(''))  # Empty string
print("transcribe('acgtxyz') should be 'ugca' <->", transcribe('acgtxyz'))  # 'ugca' (ignores unexpected characters)
print("transcribe('ACGT') should be '' <->", transcribe('ACGT'))       # '' (case-sensitive, no transcription for uppercase)
# model them from the above tests...



#
# here, write your letterScore function
def letterScore(c):
    """Returns the Scrabble score for a single-character string c."""
    # Convert to lowercase to handle both uppercase and lowercase letters
    c = c.lower()

    if c in "aeioulnstr":
        return 1
    elif c in "dg":
        return 2
    elif c in "bcmp":
        return 3
    elif c in "fhvwy":
        return 4
    elif c == "k":
        return 5
    elif c in "jx":
        return 8
    elif c in "qz":
        return 10
    else:
        return 0  # Non-letter characters score 0

# Tests
print("letterScore('A') should be 1 <->", letterScore('A'))  # 1
print("letterScore('d') should be 2 <->", letterScore('d'))  # 2
print("letterScore('B') should be 3 <->", letterScore('B'))  # 3
print("letterScore('y') should be 4 <->", letterScore('y'))  # 4
print("letterScore('k') should be 5 <->", letterScore('k'))  # 5
print("letterScore('J') should be 8 <->", letterScore('J'))  # 8
print("letterScore('Q') should be 10 <->", letterScore('Q'))  # 10
print("letterScore('1') should be 0 <->", letterScore('1'))  # 0
print("letterScore('@') should be 0 <->", letterScore('@'))  # 0
print("letterScore('') should be 0 <->", letterScore(''))    # 0 (empty string)


# be sure to try these tests!

print( "letterScore('h') should be  4 :",  letterScore('h') )
print( "letterScore('c') should be  3 :",  letterScore('c') )
print( "letterScore('a') should be  1 :",  letterScore('a') )
print( "letterScore('z') should be 10 :",  letterScore('z') )
print( "letterScore('^') should be  0 :",  letterScore('^') )


#
# here, write your scabbleScore function
#
def scrabbleScore(S):
    """Returns the Scrabble score for a string S."""
    total_score = 0
    for c in S:
        total_score += letterScore(c)
    return total_score

# Tests
print("scrabbleScore('HELLO') should be 8 <->", scrabbleScore('HELLO'))  # 8
print("scrabbleScore('SCRABBLE') should be 14 <->", scrabbleScore('SCRABBLE'))  # 14
print("scrabbleScore('PYTHON') should be 14 <->", scrabbleScore('PYTHON'))  # 14
print("scrabbleScore('QUIZ') should be 22 <->", scrabbleScore('QUIZ'))  # 22
print("scrabbleScore('123@#') should be 0 <->", scrabbleScore('123@#'))  # 0
print("scrabbleScore('') should be 0 <->", scrabbleScore(''))  # 0 (empty string)






# be sure to try these tests!

#
# Tests

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
    """Replaces all occurrences of 'a' or 'A' in the string s with the '@' symbol."""
    transformed_string = ''
    for char in s:
        if char == 'a' or char == 'A':
            transformed_string += '@'
        else:
            transformed_string += char
    return transformed_string

# Tests
print("transform('apple') should be '@pple' <->", transform('apple'))  # '@pple'
print("transform('Banana') should be 'B@n@n@' <->", transform('Banana'))  # 'B@n@n@'
print("transform('Hello World!') should be 'Hello World!' <->", transform('Hello World!'))  # 'Hello World!'




#
# Here is space to write pigletLatin and pigLatin!
#



# be sure to try these tests!
# print( "pigLatin('string')             should be  'ingstray'   :",  pigLatin('string') )
# print( "pigLatin('yttrium')            should be  'yttriumway' :",  pigLatin('yttrium') )
# print( "pigLatin('yoohoo')             should be  'oohooyay'   :",  pigLatin('yoohoo') )
# print( "pigLatin('stymie')             should be  'ymiestay'   :",  pigLatin('stymie') )


