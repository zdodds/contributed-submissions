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


#our example
c = "claremont"
answer_1 = c[8] + c[4] + c[2] + c[5] # using positive
answer_2 = c[-1] + c[-5] + c[-7] +c[-4] # using Negative
answer_3 = c[-1] + c[4] + c[-7] + c[5] # Using MIX

print("answer positive is :", answer_1)
print("answer Negative  is :", answer_2)
print("answer mix operation is :", answer_3)





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
print("Assembling care from c: ", c[0] + c[2] + c[3] + c[4])
# Example from s:
print("Assembling price from s: ", s[-2] + s[2:4] + s[1] + p[-2])

# Example from s:
print("Assembling script from s: ", s[0:5]  )

# Example from p:
print("Assembling prize from p: ", p[0] + p[-1] + p[1] + p[-3] + p[-2])

# Example from h:
print("Assembling have from h: ", h[0] + h[1] + h[-3] + h[-2])
# Example from g:
print("Assembling gard from g: ", g[0] + g[2] + g[1] + g[3])

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
print("Creating grave: ", g[0:3] + h[3:5])

# Example 3:
print("Creating clip: ", c[0:2] + s[3:5])

# Example 4:
print("Creating scrape: ", s[0:3] +h[1] +p[0:5:4])

# Example 5:
print("Creating chart: ", c[0] + h[0:3] + p[2])



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


dbl(42)  # run


# Call dbl

dbl(23)


# call dbl with a string

dbl('wow ')


#our example using print()
# This cell defines a function named   dbl

def Mohammed(F):  #function signature
    """ Result: dbl returns twice its argument
        Argument x: a number (int or float)
    """      #Docstring
    return 22*F     #return value


def IST341_Participant_5(X):
    """ Result: dbl returns twice its argument
        Argument x: a number (int or float)
    """      #Docstring
    return 22*X     #return value

print(Mohammed(2))

print(IST341_Participant_5(3))





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
    """ Return value: tpl returns thrice its argument
        Argument x: a number (int or float)
    """
    return x*x


print("sq(10) should be 100 <->", sq(10))
print("sq(46) should be 2116 <->", sq(46))


#
# your checkends function
def checkends(s):
    """
    Input a string s and returns True if the first and last character are the same.

    Argument:
    s -- a string (non-empty)

    Returns:
    True if the first and last character are the same, False otherwise.
    """
    return s[0] == s[-1]




# be sure these four tests work!  Comment them in:

print("checkends('no match') should be False <->", checkends('no match'))
print("checkends('hah! a match') should be True <->", checkends('hah! a match'))

print("checkends('q') should be True <->", checkends('q'))
print("checkends(' ') should be True <->", checkends(' '))  # single space


#
# your checkends function
def checkends(s):
    """
    Input a string s and returns True if the first and last character are the same.

    Argument:
    s -- a string (non-empty)

    Returns:
    True if the first and last character are the same, False otherwise. by using if and else
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


# be sure to digest these examples of len

print("len('claremont') should be 9 <->", len('claremont'))  # 9 characters
print("len('this is a really long string') should be 28 <->", len('this is a really long string'))
print()

print("len('q') should be 1 <->", len('q'))    # one character, 'q'
print("len(' ') should be 1 <->", len(' '))    # single space, len is 1
print()

print("len('') should be 0 <->", len(''))    # single space, len is 0
# Another Examples
print("len('python') should be 6 <->", len('python'))  # 'python' has 6 characters
print("len('Hello, Professor!') should be 17 <->", len('Hello, Professor!'))  # Including spaces and punctuation



#
# your flipside function
def flipside(s):
    m = len(s) // 2  # Find the midpoint of the string
    return s[m:] + s[:m]  # Swap the two halves and return the new string



# be sure these tests work!  Comment them in:

print("flipside('homework') should be 'workhome' <->", flipside('homework'))
print("flipside('poptart') should be 'tartpop' <->", flipside('poptart'))    # shorter first half
print("flipside('carpet') should be 'petcar' <->", flipside('carpet'))


#Using mis instead of m
#Define the function flipside(s)
def flipside(s):
    mid = len(s) // 2  # Calculate the midpoint of the string
    return s[mid:] + s[:mid]  # Swap the two halves and return the new string


#
# your transcribe_one function

# Define the function transcribe_one(s)
def transcribe_one(s):
    """
    Transcribes a single lowercase letter .

    Argument:
    s -- string a single lowercase letter (a, c, g, t) or any other string

    Returns:
    The corresponding letter if s is 'a', 'c', 'g', or 't'.
    Returns an empty string if s is anything else.
    """
    if len(s) != 1:  # Check if s is not a single character
        return ""

    if s == 'a':
        return 'u'  # Convert 'a' to 'u '
    elif s == 'c':
        return 'g'  # Convert 'c' to 'g'
    elif s == 'g':
        return 'c'  # Convert 'g' to 'c'
    elif s == 't':
        return 'a'  # Convert 't' to 'a'

    return ""  # Return an empty string for any other input



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
    """
    Converts a seconds into days, hours, minutes, and remaining seconds.

    Argument:
    s -- is a string nonnegative integer representing total seconds

    Returns:
    A list [days, hours, minutes, seconds]
    """
    days = s // 86400  # Calculate number of days (1 day = 86400 seconds)
    s = s % 86400  # Remaining seconds after extracting days

    hours = s // 3600  # Calculate number of hours (1 hour = 3600 seconds)
    s = s % 3600  # Remaining seconds after extracting hours

    minutes = s // 60  # Calculate number of minutes (1 minute = 60 seconds)
    seconds = s % 60  # Remaining seconds after extracting minutes

    return [days, hours, minutes, seconds]  # Return the list




# Try these tests!
print("convertFromSeconds(610) should return [0, 0, 10, 10] <->", convertFromSeconds(610))
print("convertFromSeconds(100000) should return [1, 3, 46, 40] <->", convertFromSeconds(100000))
#


# we do it as you request above
def convertFromSeconds(s):
    days = s // (24 * 60 * 60)  # Number of days
    s = s % (24 * 60 * 60)      # The leftover

    hours = s // (60 * 60)      # Number of hours
    s = s % (60 * 60)           # The leftover

    minutes = s // 60           # Number of minutes
    seconds = s % 60            # The leftover

    return [days, hours, minutes, seconds]  # Return as a list
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
    return s #if the string is empty, return it as is
  elif s[0] == "i":
    return "!" + replace_i(s[1:])  #if the first character is 'i', replace it with '!' and call the function on the remaining substring
  else:
    return s[0] + replace_i(s[1:])   #if the first character is not 'i', keep it unchanged and call the function on the remaining substring

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
print( "keepvwl('Mohammed') should be oae <-> ", keepvwl('Mohammed') )
print( "keepvwl('IST341_Participant_5') should be aa <-> ", keepvwl('IST341_Participant_5') )



print( "dropvwl('Mohammed') should be Mhmmd <-> ", dropvwl('Mohammed') )
print( "dropvwl('IST341_Participant_5') should be Fhd <-> ", dropvwl('IST341_Participant_5') )



#
# A full transcribe function
def transcribe_one(s):
    """ input: a string s
        output: a new string with transcribed characters
    """
    if s == "":
        return s                      #if the string is empty, return it as is
    elif s[0] == "a":
        return "u" + transcribe_one(s[1:])#replace 'a' with 'u' and call function on the remaining
    elif s[0] == "c":
        return "g" + transcribe_one(s[1:]) #replace 'c' with 'g' and call function on the remaining
    elif s[0] == "g":
        return "c" + transcribe_one(s[1:])  #replace 'g' with 'c' and call function on the remaining
    elif s[0] == "t":
        return "a" + transcribe_one(s[1:]) #replace 't' with 'a' and call function on the remaining
    else:
        return "_" + transcribe_one(s[1:])    #replace non-transcribable characters with '_'

# If you work on this, please create 4-6 tests showing how it does:

# tests here...
print( "transcribe_one('IST341_Participant_5') should be _u_u_  <->", transcribe_one('IST341_Participant_5') )
print( "transcribe_one('mohammed') should be ___u____  <->", transcribe_one('mohammed') )
print("transcribe_one('acgts') should be     ugca_     <->", transcribe_one('acgts'))
print("transcribe_one('cgu') should be gc_  <->", transcribe_one('cgu'))


# model them from the above tests...



#
# here, write your letterScore function
def letterScore(s):
    """
    Input:  single-character string s
    Output: the Scrabble score for that letter
    """
    if len(s) != 1:
        return 0        #If input is not a single character, return 0

    if s in "aeioulnstrAEIOULNSTR":
        return 1          #return letters score 1 point

    elif s in "dgDG":
        return 2      #return letters score 2 point

    elif s in "bcmpBCMP":
        return 3  #return letters score 3 points

    elif s in "fhvwyFHVWY":
        return 4      #return letters score 4 points

    elif s in "kK":
        return 5          #return letters score 5 points

    elif s in "jxJX":
        return 8  #return letters score 8 points

    elif s in "qzQZ":
        return 10      #return letters score 10 points

    else:
        return 0  #non-letter , punctuation. and numbers return score 0 point



# be sure to try these tests!

print( "letterScore('h') should be  4 :",  letterScore('h') )
print( "letterScore('c') should be  3 :",  letterScore('c') )
print( "letterScore('a') should be  1 :",  letterScore('a') )
print( "letterScore('z') should be 10 :",  letterScore('z') )
print( "letterScore('^') should be  0 :",  letterScore('^') )


#
# here, write your scabbleScore function
#

def scrabbleScore(s):
    """
    Input: a string s
    Output: the total Scrabble score of s
    """
    if s == "":
        return 0  #an empty string has a score of 0
    else:
        return letterScore(s[0]) + scrabbleScore(s[1:])  #Get score of first letter, then call the function again for the rest






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
    Input:  string s
    Output: a new string where all 'a' characters are replaced with '@'
    """
    if s == "":
        return s #If the string is empty, return it as is
    elif s[0] == "a":
        return "@" + transform(s[1:]) #transfer 'a' to '@' and call the function again for the remain
    else:
        return s[0] + transform(s[1:])  #leave the character and call the function again for the remain

print("transform('apple') should be @pple  <->", transform('apple'))
print("transform('banana') should be b@n@n@  <->", transform('banana'))
print("transform('cat') should be c@t  <->", transform('cat'))
print("transform('hello') should be hello  <->", transform('hello'))  # No 'a' to replace
print("transform('aAaA') should be @@@@  <->", transform('aAaA'))  # Case-sensitive: only lowercase 'a' is replaced


#
# Here is space to write pigletLatin and pigLatin!

def pigletLatin(s):
    if s == "":
        return ""
    vowels = "aeiou"
    if s[0] in vowels:
        return s + 'way'
    else:
        return s[1:] + s[0] + 'ay'

def pigLatin(s, index=0):
    if s == "":
        return ""
    vowels = "aeiou"
    if s[0] in vowels:
        return s + 'way'
    else:
        # Handle 'y' as consonant or vowel
        if s[0] == 'y':
            if len(s) > 1 and s[1] in vowels:
                # 'y' is consonant
                return s[1:] + 'yay'
            else:
                # 'y' is vowel
                return s + 'way'
        else:
            #if we reach the end of the string
            if index >= len(s):
                return s + 'ay'
            if s[index] in vowels or s[index] == 'y':   # If current character is a vowel or 'y', move all consonants before it to the end

                return s[index:] + s[:index] + 'ay'
            # Recursive case: move to the next character
            return pigLatin(s, index + 1)


# be sure to try these tests!
print( "pigLatin('string')             should be  'ingstray'   :",  pigLatin('string') )
print( "pigLatin('yttrium')            should be  'yttriumway' :",  pigLatin('yttrium') )
print( "pigLatin('yoohoo')             should be  'oohooyay'   :",  pigLatin('yoohoo') )
print( "pigLatin('stymie')             should be  'ymiestay'   :",  pigLatin('stymie') )


