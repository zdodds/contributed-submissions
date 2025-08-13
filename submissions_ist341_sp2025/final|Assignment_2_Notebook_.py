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
print("Assembling clear from c: ", c[0]+ c[1]+ c[4]+ c[2]+ c[3] )

print("Assembling atom from c: ", c[2]+ c[-1]+ c[-3]+ c[-4] )

print("Assembling team from c: ", c[-1]+ c[4]+ c[2]+ c[5] )

print("Assembling rental  from c: ", c[3]+ c[4]+ c[-2]+ c[-1]+ c[2]+ c[1] )

print("Assembling romance  from c: ", c[3]+ c[-3]+ c[-4]+ c[2]+ c[-2]+ c[0]+ c[4] )

# Example from g:
print("Assembling great  from g: ", g[0]+ g[1]+ g[-1]+ g[2]+ g[-2] )


s = "scripps"
# Example from s:
print("Assembling price from s: ", s[-2] + s[2:4] + s[1] + p[-2])

print("Assembling pics from s: ", s[-2] + s[3] + s[1] + s[0])

print("Assembling scrip from s: ", s[:5])

p = "pitzer"
# Example from p:
print("Assembling trip from p: ", p[2] + p[-1] + p[1] + p[0])

h = "harvey"
# Example from h:
print("Assembling heavy from h: ", h[0] + h[-2] + h[1] + h[3] + h[-1])

# Example from c and g:
c = "claremont"
g = "graduate"
print("Assembling rectangular from c and g: ", c[3] + g[-1] + c[0] + c[-1] + c[2] + c[-2] + g[0] + g[4] + c[1] + g[2] + c[3] )

# Example from s, p and h:
s = "scripps"
p = "pitzer"
h = "harvey"
print("Assembling hyperactive from s, p and h: ", h[0] + h[-1] + p[0] + p[-2] + s[2] + h[1] + s[1] + p[2] + s[3] + h[-3] + p[-2] )




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
print("Creating very clear script: ", h[3:5] + h[2] + h[-1] + c[:2] + g[-1] + c[2:4] + s[:5] + p[2])
# Example 3:
print("Creating hard luck: ", h[:3] + g[3] + c[1] + g[4] + m[1:3])
# Example 4:
print("Creating ken ate lemon: ", m[2:5] + g[5:] + c[1] + c[4:8] )
# Example 5:
print("Creating grape pie: ", g[:3] + s[4] + g[-1] + p[0:2] + p[-2])


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

dbl(5)



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
    """
    Return the square of the input argument x.
    Argument x: a number (int or float)
    """
    return x * x



# be sure these two tests work!
# you'll need to "comment them back in":

print("sq(10) should be 100 <->", sq(10))
print("sq(46) should be 2116 <->", sq(46))
print("sq(12) should be 144 <->", sq(12))
print("sq(-10) should be 100 <->", sq(-10))
print("sq(0.5) should be 0.25 <->", sq(0.5))


#
# your checkends function
def checkends(s):
    """
    Returns True if the first and last characters of the string s are the same.
    Otherwise, returns False.
    """
    if s[0] == s[-1]:
        return True
    else:
        return False





# be sure these four tests work!  Comment them in:

print("checkends('no match') should be False <->", checkends('no match'))
print("checkends('hah! a match') should be True <->", checkends('hah! a match'))

print("checkends('q') should be True <->", checkends('q'))
print("checkends(' ') should be True <->", checkends(' '))  # single space


def checkends_noif(s):
    """
    Alternate Implementation Without if/else.
    Returns True if the first and last characters of the string s are the same.
    Otherwise, returns False.
    """
    return s[0] == s[-1]


print("checkends('no match') should be False <->", checkends_noif('no match'))
print("checkends('hah! a match') should be True <->", checkends_noif('hah! a match'))

print("checkends('q') should be True <->", checkends_noif('q'))
print("checkends(' ') should be True <->", checkends_noif(' '))  # single space



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
    """
    flipside function will return a string whose first half is the second half of s
    and whose second half is the first half of s.
    """
    middle = len(s) // 2  # Determine the midpoint of the string
    return s[middle:] + s[:middle]




# be sure these tests work!  Comment them in:

print("flipside('homework') should be 'workhome' <->", flipside('homework'))
print("flipside('poptart') should be 'tartpop' <->", flipside('poptart'))    # shorter first half
print("flipside('carpet') should be 'petcar' <->", flipside('carpet'))
print("flipside('HelloWorld') should be 'WorldHello' ->", flipside('HelloWorld'))
print("flipside('12345') should be '34512' ->", flipside('12345'))


#
# your transcribe_one function
def transcribe_one(s):
    """
    This function will takes the string s and returns:
       - 'u' if s == 'a'
       - 'g' if s == 'c'
       - 'c' if s == 'g'
       - 'a' if s == 't'
       - '' (empty string) if s is any other string or if its length is not 1
    """
    if len(s) != 1:
        return ""
    elif s == 'a':
        return 'u'
    elif s == 'c':
        return 'g'
    elif s == 'g':
        return 'c'
    elif s == 't':
        return 'a'
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



#
# here, write convertFromSeconds(s)
def convertFromSeconds(s):
    """
    Takes a nonnegative integer number of seconds `s` and returns a list
    [days, hours, minutes, seconds], representing that duration.
    """
    days = s // (24 * 60 * 60)  # Number of days
    s = s % (24 * 60 * 60)      # The leftover
    hours = s // (60 * 60)      # Number of hours
    s = s % (60 * 60)           # Remaining seconds after calculating hours
    minutes = s // 60           # Number of minutes
    seconds = s % 60            # Remaining seconds

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




# fun - functions continue to exist in other cells
# try it:

replace_i('aliiien')   # a three-i'ed alien: exciting!!!


#
# A full transcribe function
def transcribe(s):
    """
    The function will takes the  string s and returns a new string by replacing some characters as below:
    - 'a' -> 'u',
    - 'c' -> 'g',
    - 'g' -> 'c',
    - 't' -> 'a'.
    - Non-transcribable characters are dropped.
    """
    if s == "":  #if the string is empty, return an empty string
        return ""

    # Get the transcription for the first character
    first_transcribed = transcribe_one(s[0])

    # Recursively transcribe the rest of the string
    return first_transcribed + transcribe(s[1:])



# If you work on this, please create 4-6 tests showing how it does:

# tests here...
# model them from the above tests...

# Test cases
print("transcribe('acgt') should be 'ugca': ", transcribe('acgt'))
print("transcribe('aaaggttcc') should be 'uuuccaagg': ", transcribe('aaaggttcc'))
print("transcribe('xacy') should be 'ug': ", transcribe('xacy'))
print("transcribe('') should be '': ", transcribe(''))
print("transcribe('gggg') should be 'cccc': ", transcribe('gggg'))
print("transcribe('tcatgc') should be 'aguacg': ", transcribe('tcatgc'))
print("transcribe('python') should be 'a': ", transcribe('python'))



#
# here, write your letterScore function
def letterScore(c):
    """
    The function takes single-character string c and returns its Scrabble Score.
    Returns 0 for non-alphabet characters or invalid input.
    """
    c = c.lower()  # Convert to lowercase

    if c in "aeioulnstr":
        return 1
    elif c in "dg":
        return 2
    elif c in "bcmp":
        return 3
    elif c in "fhvwy":
        return 4
    elif c in "k":
        return 5
    elif c in "jx":
        return 8
    elif c in "qz":
        return 10
    else:
        return 0  # For any other character, return 0




# be sure to try these tests!

print( "letterScore('h') should be  4 :",  letterScore('h') )
print( "letterScore('c') should be  3 :",  letterScore('c') )
print( "letterScore('a') should be  1 :",  letterScore('a') )
print( "letterScore('z') should be 10 :",  letterScore('z') )
print( "letterScore('^') should be  0 :",  letterScore('^') )


#
# here, write your scabbleScore function
def scrabbleScore(s):
    """
    Takes string s and returns the total Scrabble score for that string.
    """
    if s == "":  # if the string is empty, return 0
        return 0

    return letterScore(s[0]) + scrabbleScore(s[1:])


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
    """
    Recursively transforms the string s:
    - Keeps vowels from s using "keepvwl".
    - Replaces non-vowels with '_'.
    """
    if s == "":
        return ""


    first_char = s[0]
    if first_char in keepvwl(s):
        return first_char + transform(s[1:])
    else:
        return "_" + transform(s[1:])


# your tests here:
# Test cases for transform
print("transform('helloworld') should be '_e__o_o___':", transform('helloworld'))
print("transform('Python') should be '____o_':", transform('Python'))
print("transform('aeiou') should be 'aeiou':", transform('aeiou'))
print("transform('123abc!') should be '___a___':", transform('123abc!'))



#
# Here is space to write pigletLatin and pigLatin!
#


def pigletLatin(s):
    """
    The function receives single lowercase word s "string":
    - If the string is empty, return an empty string.
    - If the string starts with a vowel, add "way" to the end.
    - If the string starts with a consonant, move the first letter to the end and add "ay".
    """
    vowels = "aeiou"

    if s == "":
        return ""
    elif s[0] in vowels:
        return s + "way"
    else:
        return s[1:] + s[0] + "ay"

print("pigletLatin('one') should return 'oneway':", pigletLatin('one'))
print("pigletLatin('be') should return 'ebay':", pigletLatin('be'))
print("pigletLatin('string') should return 'tringsay':", pigletLatin('string'))
print("pigletLatin('') should return '':", pigletLatin(''))




# be sure to try these tests!
# print( "pigLatin('string')             should be  'ingstray'   :",  pigLatin('string') )
# print( "pigLatin('yttrium')            should be  'yttriumway' :",  pigLatin('yttrium') )
# print( "pigLatin('yoohoo')             should be  'oohooyay'   :",  pigLatin('yoohoo') )
# print( "pigLatin('stymie')             should be  'ymiestay'   :",  pigLatin('stymie') )





cd command-line-treasure-hunt








cd command-line-treasure-hunt








cd garden





cat README.md


