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
print("Assembling clare from c ", c[0:5])

# Example from s:
print("Assembling price from s: ", s[-2] + s[2:4] + s[1] + p[-2])

# Example from s:
print("Assembling crips from s: ", s[1:5]+s[6])

# Example from p:
print("Assembling zipper from z: ", p[3] + p[1]+ p[0]+p[0] + p[4: ])

# Example from h:
print("Assembling haha from h: ", 2*(h[0:2]))

p = "pitzer"
h = "harvey"
#    012345        for reference
print("Assembling tire from p: ", p[2:0:-1]  + p[-1:-3:-1] )
print("Assembling guard from g: ", g[0] + g[4:6]+g[1:4:2])


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
print("Creating pint: ", p[0:2] + c[7:])

# Example 2:
print("Creating dual: ", g[3:6] + c[1])

# Example 3:
print("Creating month: ", c[5: ]+ h[0])
# Example 4:
print("Creating kenny: ", m[2:6] + h[-1])
# Example 5:
print("Creating harvest: ", h[0:-1] + s[-1]+ c[-1])


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


def dbl(x):
  return 2*x
help(dbl)


def tpl(x):
    """ Return value: tpl returns thrice its argument
        Argument x: a number (int or float)
    """
    return 3*x


# Two tests are provided:
print("tpl(10) should be 30 <->", tpl(10))
"""Prints tripple of 10 as an INT, bellow it prints tpl of 10 as well, but as a
   float
"""
print("tpl(46/3) should be 46 <->", tpl(46/3))


#
# space for your sq function  (use the previous examples to get started!)

def sq(x):
    """ Return value: tpl returns double its argument
        Argument x: a number (int or float)
    """
    return x*x



# be sure these two tests work!
# you'll need to "comment them back in":

print("sq(10) should be 100 <->", sq(10))
print("sq(46) should be 2116 <->", sq(46))


def checkends(s):
  return s[0] == s[-1] if len(s) > 0 else False

#the AI did most of it once i earased it, just had to add the if statement to make the empty string work




# be sure these four tests work!  Comment them in:

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


def flipside(s):
    """ returns string in half. 1st half is s, second half is s
    """
    return s[len(s)//2:] + s[:len(s)//2]
#AI on this program gave this to me



# be sure these tests work!  Comment them in:

print("flipside('homework') should be 'workhome' <->", flipside('homework'))
print("flipside('poptart') should be 'tartpop' <->", flipside('poptart'))    # shorter first half
print("flipside('carpet') should be 'petcar' <->", flipside('carpet'))


def transcribe_one(s):
    transciption = {'a': 'u', 'c': 'g', 'g': 'c', 't': 'a'}
    if len(s) == 1 and s in transciption:
        return transciption[s]
    else:
        return ""





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
  days = s // (24*60*60)  # Number of days
  s = s % (24*60*60)      # The leftover
  hours = s // (60*60)
  s=s % (60*60)
  minutes = s // 60
  seconds = s % 60
  return [days, hours, minutes, seconds]




# Try these tests!
print("convertFromSeconds(610) should return [0, 0, 10, 10] <->", convertFromSeconds(610))
print("convertFromSeconds(100000) should return [1, 3, 46, 40] <->", convertFromSeconds(100000))
#


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




def replace_i(s):
  """ input: a string s
      output: a new string like s, with !'s
  """
  if s == "":
    return s
  elif s[0] == "i":
    return "!" + replace_i(s[1:])
  else:
    return s[0] + replace_i(s[1:])

replace_i('aliiien')   # a three-i'ed alien: exciting!!!

def replace_x(m):
    if m == "aliiien":
      return "A three-i'ed alien"
    else:
      return "Not a three-i'ed alien"

#not sure if i understood the intent, so i did both


def transcribe_one(s):
    transciption = {'a': 'u', 'c': 'g', 'g': 'c', 't': 'a'}
    return transciption[s] if s in transciption else "_"

def transcribe(s):
  if s == "":
    return ""
  return transcribe_one(s[0]) + transcribe(s[1:])

print(transcribe('claremont'))
print(transcribe('aiiiiiieee'))
print(transcribe('icily'))
print(transcribe('alien'))
print(transcribe(''))
print(transcribe('cat'))

# If you work on this, please create 4-6 tests showing how it does:

# tests here...
# model them from the above tests


def letterScore(s):
  scrabble_scores = {
      1: {'A', 'E', 'I', 'O', 'U', 'L', 'N', 'R', 'S', 'T'},
      2: {'D','G'},
      3: {'B','C','M', 'P'},
      4: {'F','H','V','W','Y'},
      5: {'K'},
      8: {'J','X'},
      10: {'Q', 'Z'}
  }

  s = s.upper()

  for score, letters in scrabble_scores.items():
    if s in letters:
        return score
  return 0


# be sure to try these tests!

print( "letterScore('h') should be  4 :",  letterScore('h') )
print( "letterScore('c') should be  3 :",  letterScore('c') )
print( "letterScore('a') should be  1 :",  letterScore('a') )
print( "letterScore('z') should be 10 :",  letterScore('z') )
print( "letterScore('^') should be  0 :",  letterScore('^') )


def scrabbleScore(s):
  scrabble_scores = {
      1: {'A', 'E', 'I', 'O', 'U', 'L', 'N', 'R', 'S', 'T'},
      2: {'D','G'},
      3: {'B','C','M', 'P'},
      4: {'F','H','V','W','Y'},
      5: {'K'},
      8: {'J','X'},
      10: {'Q', 'Z'}
  }

  s = s.upper()

  total_score = 0
  for char in s:
    if char.isalpha():
      score = letterScore(char)
      total_score += score
  return total_score

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

def transform(s, upper=True):
  if s == "":
    return ""
  first_char = s[0].upper() if upper else s[0].lower()
  return first_char + transform(s[1:], not upper)
  # this is not complete!!

print(transform("transform"))
print(transform("aiiiiiieee"))
print(transform("icily"))
print(transform("alien"))
print(transform(""))
print(transform("cat"))

# your tests here:


#
# Here is space to write pigletLatin and pigLatin!
#



# be sure to try these tests!
# print( "pigLatin('string')             should be  'ingstray'   :",  pigLatin('string') )
# print( "pigLatin('yttrium')            should be  'yttriumway' :",  pigLatin('yttrium') )
# print( "pigLatin('yoohoo')             should be  'oohooyay'   :",  pigLatin('yoohoo') )
# print( "pigLatin('stymie')             should be  'ymiestay'   :",  pigLatin('stymie') )


