perc = 0.99

if perc > 0.95:
  if perc == 9:
    print("0")
  elif perc > 9:
    print('A-')
else:
  print('Aargh!')




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
c = "claremont" #romance
g = "graduate" #trudge
m = "mckenna" #cake         # we'll skip m
s = "scripps" #crisp
p = "pitzer" #tripe
h = "harvey" #heavy
#    012345        for reference

# Your challenge: assemble one four-or-more letter word from each
# See if you can find a six letter word for one or more of them?!

# Example from c: real
print("Assembling real from c: ", c[3] + c[4] + c[2] + c[1])

# Example from c:    # find a different one from c:

# Example from s:
print("Assembling price from s: ", s[-2] + s[2:4] + s[1] + p[-2])

# Example from s:

# Example from p:

# Example from h:

p = "pitzer"
h = "harvey"
#    012345        for reference
print("Assembling tire from p: ", p[2:0:-1]  + p[-1:-3:-1] )


#    012345678     for reference
c = "claremont" #romance
g = "graduate" #trudge
m = "mckenna" #cake
s = "scripps" #crisp
p = "pitzer" #tripe
h = "harvey" #heavy
#    012345

print('Assembling romance from c:', c[3] + c[6:4:-1] + c[2] + c[7] + c[0] + c[4])
print('Assembling trudge from g:', g[6] + g[1] + g[4:2:-1] + g[0] + g[7])
print('Assembling cake from m:', m[1] + m[6] + m[2:4])
print('Assembling crisp from s:', s[1:4] + s[-1:-3:-1])
print('Assembling tripe from p:', p[2] + p[5] + p[1: :-1] + p[4])
print('Assembling heavy from h:', h[0] + h[4] + h[1] + h[3] + h[5])


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
print("Creating criterion:", s[1:4] + c[8] + p[4:] + p[1] + c[6:8])

# Example 2:
print('Creating adore:', g[2:4] + c[-3] + c[3:5])

# Example 3:
print('Creating cleaner:', c[:2] + g[-1] + m[:-3:-1] + p[4:])

# Example 4:
print('Creating trick:', c[8] + s[2:4] + m[1:3])

# Example 5:
print('Creating grave:', g[:3] + h[3:5])



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
  """ Return value: sq returns the square of its argument
      Argument x: a number (int or float)
  """
  return x**2




# be sure these two tests work!
# you'll need to "comment them back in":

print("sq(10) should be 100 <->", sq(10))
print("sq(46) should be 2116 <->", sq(46))


#
# your checkends function
def checkends(s):
  """ Return value: True if beginning and end of its argument match, False if not
      Argument: a string or number
  """
  if s[0] == s[-1]:
    return "True"
  else:
    return 'False'




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


#
# your flipside function
def flipside(s):
  """ Return value: split its argument in half and flip them around
      Argument: string
  """
  hl = len(s)//2
  return s[hl:] + s[:hl]



# be sure these tests work!  Comment them in:

print("flipside('homework') should be 'workhome' <->", flipside('homework'))
print("flipside('poptart') should be 'tartpop' <->", flipside('poptart'))    # shorter first half
print("flipside('carpet') should be 'petcar' <->", flipside('carpet'))


#
# your transcribe_one function
def transcribe_one(s):
  """ Return value: 'u' if argument is 'a', 'g' if 'c', 'c' if 'g', 'a' if 't', nothing if anything else
      Argument: a string
  """
  if s == 'a':
    return 'u'
  elif s == 'g':
    return 'c'
  elif s == 'c':
    return 'g'
  elif s == 't':
    return 'a'
  else:
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



#
# here, write convertFromSeconds(s)
#
def convertFromSeconds(s):
  """ Return value: [days, hours, minutes, seconds] of its argument
      Argument: an integer (# of seconds)
  """
  d = s//86400
  s = s%86400
  h = s//3600
  s = s%3600
  m = s//60
  s = s%60
  return [d, h, m, s]




# Try these tests!
print("convertFromSeconds(610) should return [0, 0, 10, 10] <->", convertFromSeconds(610))
print("convertFromSeconds(100000) should return [1, 3, 46, 40] <->", convertFromSeconds(100000))
#


#
# Example of a recursive function
#
s = '1'
print("'", s[1:], "'", sep = '')

def replace_i(s):
  """ input: a string s
      output: a new string like s, with
              all lowercase i's with !'s (exclamantion points)
  """
  if s == "":
    return s
  elif s[0] == "i":
    return "!" + replace_i(s[1:]) # s is alien -> s is lien -> s is ien ...
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


# Same thing as before eheheheheee
def transcribe_one(a):
  """ Return value: 'u' if argument is 'a', 'g' if 'c', 'c' if 'g', 'a' if 't', nothing if anything else
      Argument: a string
  """
  if a == 'a':
    return'u'
  elif a == 'g':
    return 'c'
  elif a == 'c':
    return 'g'
  elif a == 't':
    return 'a'


def transcribe(s):
  """ Replace 'a' with 'u'; 'g' -> 'c'; 'c' -> 'g'; t -> a
      Argument: a string
  """
  if s == '':
    return s
  elif s[0] in 'agct':
    return transcribe_one(s[0]) + transcribe(s[1:])
  else:
    return s[0] + transcribe(s[1:])


# If you work on this, please create 4-6 tests showing how it does:
print("transcribe('acgt') should be                 'ucga' <->", transcribe('agct'))
print("transcribe('caaat') should be               'guuua' <->", transcribe('caaat'))
print("transcribe('cult') should be                 'gula' <->", transcribe('cult'))
print("transcribe('of flowers') should be     'of flowers' <->", transcribe('of flowers'))
print("transcribe('and poptarts') should be 'und popauras' <->", transcribe('and poptarts'))
print("transcribe('I'm meee') should be         'I'm meee' <->", transcribe("I'm meee"))
# tests here...
# model them from the above tests...



#
# here, write your letterScore function
#

def letterScore(c):
  """ input: any letter
      output: its score in Scrabble
  """
  if c in 'aeioulnstrAEIOULNSTR':
    return 1
  elif c in 'dgDG':
    return 2
  elif c in 'bcmpBCMP':
    return 3
  elif c in 'fhvwyFHVWY': # that looks like a very nice keyboard smash :D
    return 4
  elif c in 'kK':
    return 5
  elif c in 'jxJX':
    return 8
  elif c in 'QZqz':
    return 10
  else:
    return 0


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
  ''' input: any string
      output: its scrabble score
  '''
  if S == '':
    return 0
  else:
    return letterScore(S[0]) + scrabbleScore(S[1:])





# be sure to try these tests!

#
# Tests
#
print( "scrabbleScore('quetzal')           should be  25 :",  scrabbleScore('quetzal') )
#  DAFFODIIIIIILLLLL narcissus ehehe
print( "scrabbleScore('jonquil')           should be  23 :",  scrabbleScore('jonquil') )
print( "scrabbleScore('syzygy')            should be  25 :",  scrabbleScore('syzygy') )
print( "scrabbleScore('?!@#$%^&*()')       should be  0 :",  scrabbleScore('?!@#$%^&*()') )
print( "scrabbleScore('')                  should be  0 :",  scrabbleScore('') )
print( "scrabbleScore('abcdefghijklmnopqrstuvwxyz') should be  87 :",  scrabbleScore('abcdefghijklmnopqrstuvwxyz') )


#
# here, write your own string-transforming function
#       be sure to use recursion and write at least three tests...

# based off of how I text (:PPP ; :DDD ; okkk ; yaaay ; noo ; etc)
# ... except I usually add more


def transform(s):
  """ input: a string
      output: the string, with k's, d's, p's and vowels tripled
  """
  if s == '':
    return s
  elif s[0] in 'kdpaeiouKDPAEOIU':
    return 3*s[0] + transform(s[1:])
  else:
    return s[0] + transform(s[1:])


# your tests here:
print("transform('cult')         should be                   'cuuult' <->", transform('cult'))
print("transform('of flowers')   should be         'ooof floooweeers' <->", transform('of flowers'))
print("transform('and poptarts') should be 'aaanddd pppoooppptaaarts' <->", transform('and poptarts'))
print("transform('sfhfjslfj')    should be                'sfhfjslfj' <->", transform('sfhfjslfj'))


#
# Here is space to write pigletLatin and pigLatin!


def pigletLatin(s):
  """ initial consonant get moved to the end and ay is added
      if first letter is a vowel, way is added to the end
  """
  if  s =='':
    return ''
  elif s[0] in 'aeiou' or s[0] in 'y' and s[1] not in 'aeiou':
    return s + 'way'
  else:
    return s[1:] + s[0] + 'ay'

print( "pigletLatin('string')     should be     'tringsay':",  pigletLatin('string') )
print( "pigletLatin('birb')       should be       'irbbay':",  pigletLatin('birb') )
print( "pigletLatin('be')         should be         'ebay':",  pigletLatin('be') )
print( "pigletLatin('')           should be             '':",  pigletLatin('') )
assert pigletLatin('assert') == 'assertway'
print()



# p i g
def pig(S,l):
  """when it's not a vowel
     piglatin when it's a consonant
     aka this is the recursive part
  """
  if S[0] in 'aeiou' or S[0] in 'y' and S[1] not in 'aeiou':
    return S + l + 'ay'
  else:
    l = l + S[0]
    return pig(S[1:], l)

def pigLatin(s):
  """ all initial consonants get moved to the end and ay is added
      if first letter is a vowel, way is added to the end
  """
  if   s =='':
    return ''
  elif s[0] in 'aeiou' or s[0] == 'y' and s[1] not in 'aeiou':
    return s + 'way'
  else:
    return pig(s[1:], s[0])

# be sure to try these tests!
print( "pigLatin('string')        should be     'ingstray':",  pigLatin('string') )
print( "pigLatin('yttrium')       should be   'yttriumway':",  pigLatin('yttrium') )
print( "pigLatin('yoohoo')        should be     'oohooyay':",  pigLatin('yoohoo') )
print( "pigLatin('stymie')        should be     'ymiestay':",  pigLatin('stymie') )





cd command-line-treasure-hunt








cat README.md





cat LICENSE


cd garden/





cat README.md





cd house/





cat README.md


cd front_room/





cat README.md


#


cd upstairs/





cat README.md


#


cd kitchen/





cat README.md


#/upstairs





cd bedroom/





cat TREASURE.md


cd bed/





cat KITTY.md








cat DOGGIE.md


# trusting the Ai...



# was i right to trust the AI?



#not sure how that works TwT


# OPTION TWO
with open('DOGGIE.md', 'w') as f:
    f.write('''You see a dog lying next to the kitty.
It is very cute.
The kitty won't be lonely anymore! Yay!''')


cat DOGGIE.md


# YAYYYYYY











#





cd closet/





cat MONSTER.md


rm MONSTER.md








#





#





#





# That was fun :D


