# single-character substitution:

def i2I_once(s):
  """ i2I_once changes an input 'i' to an output 'I'
      all other inputs remain the same
  """
  if s == 'i':
    return 'I'
  else:
    return s

# tests are in the next cell...


# Tests to try for i2I_once  (Try control-/ to uncomment them!)

print("i2I_once('i') should be 'I' <->", i2I_once('i'))
print("i2I_once('j') should be 'j' <->", i2I_once('j'))
print()

print("i2I_once('alien') should be 'alien' <->", i2I_once('alien'))
print("i2I_once('icicle') should be 'icicle' <->", i2I_once('icicle'))


# multiple-character substitution

def i2I_all(s):
  """ changes i to I for all characters in the input, s """
  if s == '':       # EMPTY case!
    return ''
  else:
    return i2I_once(s[0]) + i2I_all(s[1:])  # FIRST and REST

# tests are in the next cell...


# Tests to try for i2I_all

print("i2I_all('alien') should be 'alIen' <->", i2I_all('alien'))
print("i2I_all('aliiien') should be 'alIIIen' <->", i2I_all('aliiien'))
print()

print("i2I_all('icicle') should be 'IcIcle' <->", i2I_all('icicle'))
print("i2I_all('No eyes to see here') should be 'No eyes to see here' <->", i2I_all('No eyes to see here'))


# multiple-character substitution

def i2I_all_loop(s):
  """ changes i to I for all characters in the input, s """
  result = ''        # the result starts as the EMPTY string!

  for c in s:        # c is the name we are giving to EACH element in s (that is, each character in s)
    result = result + i2I_once(c)   # each time, we add the new transformed output to the end of result
    # print("result is", result)    # uncomment this to see the process, in action!

  return result      # at the end, we return the final, completely-created result string!

# tests are in the next cell...


# Tests to try for i2I_all_loop

print("i2I_all_loop('alien') should be 'alIen' <->", i2I_all_loop('alien'))
print("i2I_all_loop('aliiien') should be 'alIIIen' <->", i2I_all_loop('aliiien'))
print()

print("i2I_all_loop('icicle') should be 'IcIcle' <->", i2I_all_loop('icicle'))
print("i2I_all_loop('No eyes to see here') should be 'No eyes to see here' <->", i2I_all_loop('No eyes to see here'))


#
# Example of recursion-as-future:
#

from random import *

def guess( hidden ):
    """
        have the computer guess numbers until it gets the "hidden" value
        return the number of guesses
    """
    this_guess = choice( range(0,100) )  # 0 to 99, inclusive

    if this_guess == hidden:
        print("I guessed it!")
        return 1                         # it only took one guess!

    else:
        return 1 + guess( hidden )  # 1 for this guess, PLUS all future guesses!

# test our function!
guess(42)


s = 'shout!'
print("s is", s)
print("s.upper() is", s.upper())


s = 'WHISPER...'
print("s is", s)
print("s.lower() is", s.lower())


import random

def spongebobbify_each(s):
    """Returns a randomly capitalized version of the input string.

    Argument:
    s -- A string

    Returns:
    A string where each letter is randomly converted to upper or lower case.
    """
    return ''.join(random.choice([char.upper(), char.lower()]) for char in s)

# Test cases
print("spongebobbify_each(\"where's gary?\") ->", spongebobbify_each("where's gary?"))
print("spongebobbify_each(\"hello world\") ->", spongebobbify_each("hello world"))
print("spongebobbify_each(\"Python programming\") ->", spongebobbify_each("Python programming"))
print("spongebobbify_each(\"spongebob squarepants!\") ->", spongebobbify_each("spongebob squarepants!"))
print("spongebobbify_each(\"this is so random\") ->", spongebobbify_each("this is so random"))



# We provide the one-character version, in this case:
import random                  #  get the random library

def spongebobbify_once(s):
  """ returns the input, randomly "upper-cased" or "lower-cased" """
  result = random.choice( [s.upper(), s.lower()] )   # choose one at random
  return result

# Tests in the next cell...


# Tests to try for spongebobbify_once
# There are not "right" answers, so we just test them:

print(spongebobbify_once('F is for friends who do stuff together!'))
print(spongebobbify_once('I knew I shouldn\'t have gotten out of bed today.'))
print()

# but we want to use it on single letters!
print(spongebobbify_once('a'))
print(spongebobbify_once('b'))
print(spongebobbify_once('c'))
print(spongebobbify_once('d'))
print(spongebobbify_once('e'))


# Here, write your  spongebobify_all(s)
#
import random

def spongebobify_each(s):
    """Returns a randomly capitalized version of the input string.

    Argument:
    s -- A string

    Returns:
    A string where each letter is randomly converted to upper or lower case.
    """
    result = ""  # Initialize an empty result string
    for char in s:
        # Randomly decide if the character should be uppercase or lowercase
        result += random.choice([char.upper(), char.lower()])
    return result



# Try our tests (next cell) and add three of your own...
# Preferably sBoB QuoTeS...


# Tests to try for spongebobbify_once
# Tests to try for spongebobbify_once
# Tests to try for spongebobbify_once
# Tests to try for spongebobbify_once
# There are not "right" answers, so we just test them:

print("spongebobbify_all('F is for friends who do stuff together!')")
print("spongebobbify_all('I knew I shouldn\'t have gotten out of bed today.')")
print("spongebobbify_all('The inner machinations of my mind are an enigma. - Patrick')")
print("spongebobify_each(\"spongebob squarepants!\") ->", spongebobify_each("spongebob squarepants!"))
print("spongebobify_each(\"I love coding!\") ->", spongebobify_each("I love coding!"))
print("spongebobify_each(\"This is an interesting assignment.\") ->", spongebobify_each("This is an interesting assignment."))


# Your tests here:


#
# Use this cell -- and/or create more cells -- for your encode and decode functions
# There will be four functions in total!
#
def encode_each(s):
    """Encodes a string by replacing characters using ENCODE_MAP.

    Argument:
    s -- The original string

    Returns:
    The encoded string with substitutions.
    """
    result = ""
    for char in s:
        result += ENCODE_MAP.get(char.lower(), char)  # Keep unchanged if not in map
    return result

# Test encoding
print("encode_each('hello world') ->", encode_each("hello world"))


def decode_each(s):
    """Decodes a string by replacing encoded characters back to their original form.

    Argument:
    s -- The encoded string

    Returns:
    The decoded string back to its original form.
    """
    result = ""
    for char in s:
        result += DECODE_MAP.get(char, char)  # Keep unchanged if not in map
    return result

# Test decoding
encoded_msg = encode_each("hello world")
print("decode_each(encoded_msg) ->", decode_each(encoded_msg))

def encode_once(s):
    """Recursively encodes a string by replacing characters using ENCODE_MAP."""
    if s == "":
        return ""  # Base case: Empty string remains empty
    return ENCODE_MAP.get(s[0].lower(), s[0]) + encode_once(s[1:])  # Replace first, recurse on rest

# Test encoding (recursive)
print("encode_once('hello world') ->", encode_once("hello world"))

def decode_once(s):
    """Recursively decodes a string by replacing encoded characters back to their original form."""
    if s == "":
        return ""  # Base case: Empty string remains empty
    return DECODE_MAP.get(s[0], s[0]) + decode_once(s[1:])  # Replace first, recurse on rest

# Test decoding (recursive)
encoded_msg = encode_once("hello world")
print("decode_once(encoded_msg) ->", decode_once(encoded_msg))


# Our tests are below. Then, add three tests of your own:


CGU = """Claremont Graduate University prepares individuals to be leaders
for positive change in the world. Unique in its transdisciplinary approach,
the university is dedicated to the creation, dissemination, and application
of new knowledge and diverse perspectives through research, practice,
creative works, and community engagement.
"""

E = encode_each(CGU)
print("encode_all(CGU) is", E)

D = decode_each(E)
print("decode_all(E) is", D)  # should be the original!


CMC = """Claremont McKenna College's mission is to educate its students
for thoughtful and productive lives and responsible leadership in
business, government, and the professions, and to support faculty
and student scholarship that contribute to intellectual vitality
and the understanding of public policy issues."""

E = encode_each(CMC)
print("encode_all(CMC) is", E)

D = decode_each(E)
print("decode_all(E) is", D)  # should be the original!


SCR = """The mission of Scripps College is to educate women to
develop their intellects and talents through active participation
in a community of scholars, so that as graduates they may contribute
to society through public and private lives of leadership, service,
integrity, and creativity.."""

E = encode_each(SCR)
print("encode_all(SCR) is", E)

D = decode_each(E)
print("decode_all(E) is", D)  # should be the original!




#
# Above - or here - include three encode/decode tests of your own...
#
test_sentences = [
    "SpongeBob SquarePants is the best!",
    "Python programming is fun.",
    "Let's encode and decode this message."
]

# Run tests
for sentence in test_sentences:
    encoded = encode_each(sentence)
    decoded = decode_each(encoded)
    print(f"\nOriginal: {sentence}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")



def fun4():
  for i in range(1,6):
    if i%2 == 0:
      print("i is", i)
  return


#
# Here are vwl_once and vwl_all
#

def vwl_once(s):
  """ returns a score of 1 for single-character vowels aeiou
      returns a score of 0 for everything else
  """
  if len(s) != 1:    # not a single-character? score is 0
    return 0
  else:
    s = s.lower()    # simplify by making s lower case
    if s in 'aeiou':      # if s is in that string, it's a vowel: score is 1
      return 1
    else:                 # if not: score is 0
      return 0


def vwl_count(s):
  """ returns the total "vowel-score for an input string s
      that is, we return the number of vowels in s
  """
  # you need to write this one!
  # use the previous examples (especially the "each" examples) as a guide! :-)

def vwl_once(s):
    """Returns 1 if the character is a vowel, otherwise 0.

    Argument:
    s -- A single character string

    Returns:
    1 if vowel, 0 if not.
    """
    return 1 if s.lower() in 'aeiou' else 0

# Test cases for single character input
print("vwl_once('a') should be 1 <->", vwl_once('a'))
print("vwl_once('b') should be 0 <->", vwl_once('b'))
print("vwl_once('E') should be 1 <->", vwl_once('E'))
print("vwl_once('z') should be 0 <->", vwl_once('z'))
print("vwl_once('O') should be 1 <->", vwl_once('O'))


def vwl_count(s):
    """Counts the number of vowels in the input string s.

    Argument:
    s -- A string

    Returns:
    The total number of vowels in s.
    """
    if s == "":  # Base case: empty string has 0 vowels
        return 0
    return vwl_once(s[0]) + vwl_count(s[1:])  # Sum vowels recursively

# Test cases for counting vowels
print("vwl_count('hello') should be 2 <->", vwl_count('hello'))
print("vwl_count('Python') should be 1 <->", vwl_count('Python'))
print("vwl_count('aeiou') should be 5 <->", vwl_count('aeiou'))
print("vwl_count('xyz') should be 0 <->", vwl_count('xyz'))
print("vwl_count('This is a test sentence.') should be 7 <->", vwl_count('This is a test sentence.'))


# Tests and tests-to-write are in the next cells:


print("vwl_count('English usually has lots of vowels.') should be 10 <->", vwl_count('English usually has lots of vowels.'))
print("The CGU mission statement has this many vowels: (let's see!) <->", vwl_count(CGU))


#
# Part 1 task: determine which mission statement has the most vowels-per-character!
#
#        Hint: count the vowels, then use len to determine vowels-per-character!
#        Compare the three strings already defined above:   CGU,  CMC,  and SCR

def compare_vowel_count(cgu, cmc, scr):
    """Determines which mission statement has the most vowels.

    Arguments:
    cgu -- Mission statement of CGU
    cmc -- Mission statement of CMC
    scr -- Mission statement of SCR

    Returns:
    The name of the institution with the highest number of vowels.
    """
    # Count vowels in each mission statement
    cgu_vowel_count = vwl_count(cgu)
    cmc_vowel_count = vwl_count(cmc)
    scr_vowel_count = vwl_count(scr)

    # Print results
    print(f"CGU Vowel Count: {cgu_vowel_count}")
    print(f"CMC Vowel Count: {cmc_vowel_count}")
    print(f"SCR Vowel Count: {scr_vowel_count}")

    # Determine which has the most vowels
    max_vowels = max(cgu_vowel_count, cmc_vowel_count, scr_vowel_count)

    if max_vowels == cgu_vowel_count:
        return "CGU has the most vowels!"
    elif max_vowels == cmc_vowel_count:
        return "CMC has the most vowels!"
    else:
        return "SCR has the most vowels!"

# Mission statements (Replace with actual statements)
cgu_mission = "Claremont Graduate University is a graduate-only research university."
cmc_mission = "Claremont McKenna College fosters leadership, innovation, and impact."
scr_mission = "Scripps College empowers women through interdisciplinary liberal arts education."

# Compare and determine the winner
result = compare_vowel_count(cgu_mission, cmc_mission, scr_mission)
print(result)




#
# Part 2 task: determine whose prose is more vowel-rich?
# + find a paragraph of prose you've written
# + find a paragraph a friend has written (or another one that you have!)
#
# Assign each to a variable:

YOURS = """  <paste your prose here
it's ok to have multiple lines inside
triple-quoted strings>
"""

THEIRS = """  <paste _their_ prose here
again, ok to have multiple lines inside
triple-quoted strings>
"""

#
# This analysis is similar to the mission statements...
#

def vwl_count(s):
    """Counts the number of vowels in the input string s.

    Argument:
    s -- A string

    Returns:
    The total number of vowels in s.
    """
    if s == "":  # Base case: empty string has 0 vowels
        return 0
    return (1 if s[0].lower() in 'aeiou' else 0) + vwl_count(s[1:])  # Sum vowels recursively

def compare_prose_vowels(yours, theirs):
    """Determines which prose contains more vowels.

    Arguments:
    yours -- A paragraph of prose written by you
    theirs -- A paragraph of prose written by someone else

    Returns:
    The prose (YOURS or THEIRS) that contains the most vowels.
    """
    # Count vowels in both prose paragraphs
    yours_vowel_count = vwl_count(yours)
    theirs_vowel_count = vwl_count(theirs)

    # Print results
    print(f"Your Vowel Count: {yours_vowel_count}")
    print(f"Their Vowel Count: {theirs_vowel_count}")

    # Determine which has the most vowels
    if yours_vowel_count > theirs_vowel_count:
        return "Your prose has more vowels!"
    elif theirs_vowel_count > yours_vowel_count:
        return "Their prose has more vowels!"
    else:
        return "Both prose paragraphs have the same number of vowels!"

# Assign prose to variables
YOURS = """The evolution of artificial intelligence has transformed modern technology in ways that were once unimaginable.
From natural language processing to autonomous systems, AI continues to shape our future.
With each advancement, ethical considerations become increasingly important, ensuring responsible development."""

THEIRS = """Throughout history, technological progress has always been a double-edged sword.
While innovation leads to efficiency and new opportunities, it also brings challenges and disruptions.
Balancing these aspects is crucial for sustainable development and a better society."""

# Compare and determine the winner
result = compare_prose_vowels(YOURS, THEIRS)
print(result)




# this imports the library named random

import random

# once it's imported, you are able to call random.choice(L) for any sequence L
# try it:



# Try out random.choice -- several times!
result = random.choice( ['claremont', 'graduate', 'university'] )
print("result is", result)



# let's see a loop do this 10 times!

for i in range(10):                # loop 10 times
    result = random.choice( ['claremont', 'graduate', 'university'] ) # choose
    print("result is", result)     # print



#
# you can also import a library can be imported by using this line:

from random import *

# when the above line is run, you are able to call choice(L) for any sequence L
#
# note that you won't need random.choice(L)
# let's try it!



result = choice( ["rock", "paper", "scissors"] )
print("result is", result)



# Python can create lists of any integers you'd like...
L = list(range(0,100))    # try different values; try omitting/changing the 0
print(L)



# combining these, we can choose random integers from a list
result = choice( range(0,100) )   # from 0 to 99
print("result is", result)



# let's run this 10 times!
for i in range(0,10):
    result = choice( range(0,100) )   # from 0 to 99
    print("result is", result)



# let's get more comfortable with loops...

for i in [0,1,2]:     # Key: What variable is being defined and set?!
    print("i is", i)




# Note that range(0,3) generates [0,1,2]

for i in range(0,3):     # Key: What variable is being defined and set?!
    print("i is", i)

# When would you _not_ want to use range for integers?



# Usually i is for counting, x is for other things (wise, not req.)

for x in [2,15,2025]:     # Key: the loop variable
    print("x is", x)

# When would you _not_ want to use range for integers?


# How could we get this to print "Happy birthday!" 42 times?

for i in range(42):
    print('Happy birthday!')



# return _after_ a loop:

def funA():
    for i in range(0,3):
        print("i is", i)
    return

# why does this not print anything?!??
# We need to call the function in order to get a result.

def funA():
    for i in range(0, 3):
        print("i is", i)
    return  # This return is optional, as Python functions return None by default.

# Call the function to see output
funA()




# return _within_ a loop:

def funB():
    for i in range(0,3):
       print("i is", i)

# What do we need here?  (Is this what you expect?!)
funB()
#We need to call the function again and since it's a loop we don't need a return



# let's add an if statement (a conditional)
#              ... to test different indentations

def funB1():
    for i in range(1,6):
        if i%2 == 0:
            print("i is", i)
funB1()




# an add-em-up function

def addup(N):
    """ adds from 1 through N (inclusive)
    """
    result = 0

    for x in range(1,N+1):
        result = result + x

    return result
N = 4
result = addup(N)
print(f"addup({N}) should be 0+1+2+3+{N} == {result}")
# addup(4) should be 0+1+2+3+4 == 10



# an factorial function

def fac(N):
    """ a factorial function, returns the factorial of the input N
    """
    result = 1

    for x in range(1,N+1):
        result = result * x

    return result
N = 4
result = fac(N)
print(f"fac({N}) should be 1*2*3*4 == {result}")
# fac(4) should be 1*2*3*4 == 24


"""

Loops we tried in our "breakout quiz":

# upper left
result = 1
for x in [2,5,1,4]:
    result *= x
print(result)

# upper right
x = 0
for i in range(4):
    x += 10
print(x)

# lower left
L = ['golf','fore!','club','tee']
for i in range(len(L)):
    if i%2 == 1:
        print(L[i])

# lower right
S = 'time to think this over! '
result = ''
for i in range(len(S)):
    if S[i-1] == ' '
        result += S[i]
print(result)

"""



# staging area...

print("Start!")




#
# Functions with for loops to write:
#
# for loops:
#
# summer(L)               returns the sum of the #'s in L
# summedOdds(L)           returns the sum of the _odd_ #'s in L
# summedExcept(exc, L)    returns the sum of all #'s in L not equal to exc
# summedUpto(exc, L)      returns the sum of all #'s in L upto exc (not including exc)


# examples:
#       summer( [2,3,4,1] )    ->  10
#   summedOdds( [2,3,4,1] )    ->   4
# summedExcept( 4, [2,3,4,1] ) ->   6
#   summedUpto( 4, [2,3,4,1] ) ->   5


#
# here, write the summer function!
#

def summer(L):
  """ uses a for loop to add and return all of the elements in L
  """
  total = 0
  for num in L:
        total += num
  return total


# Here are two tests -- be sure to try them!
print("summer( [2,3,4,1] )  should be 10 <->", summer( [2,3,4,1] ))
print("summer( [35,3,4,100] )  should be 142 <->", summer( [35,3,4,100] ))


#
# here, write the summedOdds function!
#

def summedOdds(L):
    """ uses a for loop to add and return all of the _odd_ elements in L
    """
    total = 0
    for num in L:
        if num % 2 == 1:  # Check if the number is odd
            total += num
    return total


# Here are two tests -- be sure to try them!
print("summedOdds( [2,3,4,1] )  should be 4 <->", summedOdds( [2,3,4,1] ))
print("summedOdds( [35,3,4,100] )  should be 38 <->", summedOdds( [35,3,4,100] ))


#
# here, write the summedExcept function!
#

def summedExcept( exc, L ):
    """ include a short description here!
    """
    total = 0
    for num in L:
        if num != exc:  # Skip the exception number
            total += num
    return total


# Here are two tests -- be sure to try them!
print("summedExcept( 4, [2,3,4,1] )  should be 6 <->", summedExcept( 4, [2,3,4,1] ))
print("summedExcept( 4, [35,3,4,100] )  should be 138 <->", summedExcept( 4, [35,3,4,100] ))




#
# here, write the summedUpto function!
#

def summedUpto( exc, L ):
    """ include a short description here!
    """
    total = 0
    for num in L:
        if num == exc:  # Stop summing when 'exc' is found
            break
        total += num
    return total





# Here are two tests -- be sure to try them!
print("summedUpto( 4, [2,3,4,1] )  should be 5 <->", summedUpto( 4, [2,3,4,1] ))
print("summedUpto( 100, [35,3,4,100] )  should be 42 <->", summedUpto( 100, [35,3,4,100] ))



#
# Example while loop: the "guessing game"
#

from random import *

def guess( hidden ):
    """
        have the computer guess numbers until it gets the "hidden" value
        return the number of guesses
    """
    guess = -1      # start with a wrong guess and don't count it as a guess
    number_of_guesses = 0   # start with no guesses made so far...

    while guess != hidden:
        guess = choice( range(0,100) )  # 0 to 99, inclusive
        number_of_guesses += 1

    return number_of_guesses

# test our function!
guess(42)



#
# Functions with while loops to write:
#

# guess_between(low,high) like guess, but until it gets a number anywhere between
#                         low and high. Specifically, until it guesses
#                         less than high, and greater than or equal to low.
#
#
# listTilRepeat(high)     accumulates a list of values in range(0,high) until one repeats
#                         and returns the whole list
#


# examples (don't forget the randomness will change things!)
#
# guess_between(40,50)   ->   8    (on average, around 10)
#
# listTilRepeat(10)      ->   [4, 7, 8, 3, 7]     (the final # must be a repeat)
# listTilRepeat(10)      ->   [2, 1, 9, 9]     (the final # must be a repeat)



#
# here, write guess_between
#
import random # Import the random module
def guess_between(low,high):
    """ guesses a # from 0 to 99 (inclusive) until
        it gets one that is strictly less than high and
        greater than or equal to low
        Then, this function returns the total # of guesses made
    """

    guess = random.randint(0, 100)  # Start with a random number
    while not (low <= guess < high):  # Keep guessing until within range
        guess = random.randint(0, 100)
    return guess


#
# be sure to test your guess_between here -- and leave the test in the notebook!
#
print("guess_between(40,50) ->", guess_between(40, 50))
print("guess_between(10,20) ->", guess_between(10, 20))



# Try out adding elements to Lists

L = [3,4]
print("Before: L is", L)

guess = 42
L = L + [guess]
print(" After: L is", L)


#
# here, write listTilRepeat
#

def listTilRepeat(high):
    """ this f'n accumulates random guesses into a list, L, until a repeat is found
        it then returns the list (the final element should be repeated, somewhere)
    """
    seen = set()  # Track numbers seen so far
    numbers = []  # Store generated numbers

    while True:
        num = random.randint(0, high - 1)  # Generate a random number within range
        if num in seen:  # Stop if it's a repeat
            numbers.append(num)
            break
        seen.add(num)
        numbers.append(num)

    return numbers


#
# be sure to test your listTilRepeat here -- and leave the test in the notebook!
#
print("listTilRepeat(10) ->", listTilRepeat(10))
print("listTilRepeat(5) ->", listTilRepeat(5))
print("listTilRepeat(15) ->", listTilRepeat(15))




# The birthday paradox is the fact that
#     listTilRepeat(365) has surprisingly few elements!
#
# Run listTilRepeat(365) a few times and print its _length_ each time
#     (Don't print the lists... it's too much to digest.)
#
# To many people, the results feel counterintuitive!



#
# Example while loop: the "guessing game"
#

from random import *

def guess( hidden ):
    """
        have the computer guess numbers until it gets the "hidden" value
        return the number of guesses
    """
    guess = hidden - 1      # start with a wrong guess + don't count it as a guess
    number_of_guesses = 0   # start with no guesses made so far...

    while guess != hidden:
        guess = choice( range(0,100) )  # 0 to 99, inclusive
        number_of_guesses += 1

    return number_of_guesses

# test our function!
guess(42)



#
# Example Monte Carlo simulation: rolling two dice and counting doubles
#

from random import *
import time

def count_doubles( num_rolls ):
    """
        have the computer roll two six-sided dice, counting the # of doubles
        (same value on both dice)
        Then, return the number of doubles...
    """
    numdoubles = 0       # start with no doubles so far...

    for i in range(0,num_rolls):   # roll repeatedly: i keeps track
        d1 = choice( [1,2,3,4,5,6] )  # 0 to 6, inclusive
        d2 = choice( range(1,7) )     # 0 to 6, inclusive
        if d1 == d2:
            numdoubles += 1
            you = "🙂"
        else:
            you = " "

        print("run", i, "roll:", d1, d2, you, flush=True)
        time.sleep(.01)

    return numdoubles

# test our function!
count_doubles(10)



#
# Example Monte Carlo simulation: the Monte-Carlo Monte Hall paradox
#

import random
import time

def count_wins( N, original_choice, stay_or_switch ):
    """
        run the Monte Hall paradox N times, with
        original_choice, which can be 1, 2, or 3 and
        stay_or_switch, which can be "stay" or "switch"
        Count the number of wins and return that number.
    """
    numwins = 0       # start with no wins so far...

    for i in range(1,N+1):      # run repeatedly: i keeps track
        win_curtain = random.choice([1,2,3])   # the curtain with the grand prize
        original_choice = original_choice      # just a reminder that we have this variable
        stay_or_switch = stay_or_switch        # a reminder that we have this, too

        result = ""
        if original_choice == win_curtain and stay_or_switch == "stay": result = " Win!!!"
        elif original_choice == win_curtain and stay_or_switch == "switch": result = "lose..."
        elif original_choice != win_curtain and stay_or_switch == "stay": result = "lose..."
        elif original_choice != win_curtain and stay_or_switch == "switch": result = " Win!!!"

        print("run", i, "you", result, flush=True)
        time.sleep(.025)

        if result == " Win!!!":
            numwins += 1


    return numwins

# test our three-curtain-game, many times:
count_wins(10, 1, "stay")




# More Monte Carlo simulations!

#
# Example of a random-walk (but no animation is intended here...)
#

import random

def rs():
    """ One random step """
    return random.choice([-1, 1])


def rpos(start,N):
    """ wander from start for N steps, printing as we go
        return the position at the end (the final "current" position)
    """
    current = start        # our current position begins at start...

    for i in range(N):     # step repeatedly:  i keeps track from 0..N
        print("At location:", current)
        current = current + rs()  # add one step,

    print("At location:", current)
    return current


# let's test it, perhaps start at 47 and take 9 steps...
rpos(47,9)



# Monte Carlo simulation #2... the random-walker

#
# Task #1:  understand the _single_ "sleepwalker":
#           change the character sleepwalking to something else, emoji, etc.
#           change the "walls" to something else
#           change the "sidewalk" to something else (water, air, walls, ...)
#           try it with some different inputs... then you'll be ready for task 2

import time
import random


def rs():
    """One random step (-1 or +1)."""
    return random.choice([-1, 1])

def rwalk(start, low, high):
    """Random walk with animated display.

    Arguments:
    start -- Starting position
    low   -- Left boundary
    high  -- Right boundary

    Returns:
    The total number of steps taken before hitting a boundary.
    """
    totalsteps = 0
    current = start

    # Customize visual elements
    walker = "😴"    # Sleepwalker character
    path = "~"       # Path type (e.g., "~" for water, " " for air, "🛤" for road)
    wall = "🌲"      # Wall/boundary type (e.g., "🏰" for castle walls)

    while True:
        if current <= low:  # Left boundary reached
            return totalsteps
        elif current >= high:  # Right boundary reached
            return totalsteps
        else:
            current += rs()  # Take a step
            totalsteps += 1

            # Animate the walker
            left_side = current - low
            right_side = high - current
            print(wall + path*left_side + walker + path*right_side + wall, flush=True)
            time.sleep(0.05)  # Adjust animation speed

# Try different settings
rwalk(5, 0, 10)  # Start at 5, boundaries at 0 and 10
# rwalk(15, 0, 15)  # Uncomment to try a longer range


    # the code can never get here!




#
# Task #2:  create a _two_ sleepwalker animation!
#           For an example _idea_ see the next cell...

import time
import random

def rs():
    """One random step (-1, 0, or +1) with slight bias for movement forward."""
    return random.choices([-1, 0, 1], weights=[4, 1, 5])[0]

def random_walkers(start1, start2, low, high, steps=50):
    """Simulates two random walkers moving on a 1D path.

    Arguments:
    start1 -- Starting position of first walker
    start2 -- Starting position of second walker
    low    -- Left boundary
    high   -- Right boundary
    steps  -- Maximum number of steps

    Returns:
    None (prints animation of walkers)
    """
    walker1 = "🐾"   # Walker 1 (Animal footprints)
    walker2 = "🤖"   # Walker 2 (Robot)
    path = "·"       # Aesthetic path
    wall = "🌲"      # Walls (forest theme)

    pos1, pos2 = start1, start2  # Initialize positions

    for _ in range(steps):
        # Move walkers
        pos1 += rs()  # Random movement
        pos2 += rs()  # Random movement

        # Keep walkers inside boundaries
        pos1 = max(low, min(high, pos1))
        pos2 = max(low, min(high, pos2))

        # Create visual representation
        left_side = min(pos1, pos2) - low
        right_side = high - max(pos1, pos2)

        # Ensure characters are placed correctly
        if pos1 < pos2:
            line = wall + path * left_side + walker1 + path * (pos2 - pos1 - 1) + walker2 + path * right_side + wall
        elif pos1 > pos2:
            line = wall + path * left_side + walker2 + path * (pos1 - pos2 - 1) + walker1 + path * right_side + wall
        else:
            line = wall + path * left_side + "🔥" + path * right_side + wall  # 🔥 if they collide

        # Print and animate
        print(line, flush=True)
        time.sleep(0.1)

# Run the simulation
random_walkers(start1=5, start2=15, low=0, high=20, steps=30)




# here is an _example_ of a two-sleepwalker animation idea
# a starting point has been written... but only one wanderer is wandering!
# your task is to make sure TWO wanderers are wandering... in a fashion you design...

import time
import random

def rs():
    """ One random step """
    return random.choice([-1, 1])

def print_poptarts(pST, pSM):
    """ print the two poptarts! """
    if pST < pSM:
        pLeft = pST;   cLeft = "\033[6;33;41m" + "P" + "\033[0m"
        pRight = pSM;  cRight = "\033[6;36;43m" + "P" + "\033[0m"
    else:
        pLeft = pSM;   cLeft = "\033[6;36;43m" + "P" + "\033[0m"
        pRight = pST;  cRight = "\033[6;33;41m" + "P" + "\033[0m"

    left_space = (pLeft-0)
    middle_space = (pRight-pLeft)
    right_space = (30-pRight)

    print("CGU|" + "_"*left_space + cLeft + "_"*middle_space + cRight + "_"*right_space + "|Toaster", flush=True)


def poptart_race(pST, pSM):
    """
        This simulator observes two poptarts, pST, pSM (you can guess the flavors...)
        wandering between 0 and 30.

        Call this with
               poptart_race(10, 20)
           or  poptart_race(pST=10, pSM=20)    # this is the same as the line above

        The endpoints are always at 0 and 30. We check that  0 < pST < 30 and 0 < pSM < 30

        Other values to try:  poptart_race(18, 22)    # evenly spaced
                              poptart_race(5, 15)     # uneven spacing: pST is closer...
    """
    num_steps = 0       # count the number of steps

    while 0 < pST < 30:
        print_poptarts(pST, pSM)   # print the current poptart-configuration!
        pST = pST + rs()           # take a random step for the strawberry poptart...
        num_steps += 1             # add 1 to our count of steps (in the variable num_steps)
        time.sleep(0.05)           # pause a bit, to add drama!

    # finished with the while loop!
    return num_steps




poptart_race(10, 20)


import time

# emoji test
emoji_list = [ "♫", "♪" ]
for i in range(1,10):
    left_side = i
    right_side = (10-i)

    e = "🙂"
    # e = random.choice(emoji_list)

    print("|" + "_"*left_side + e + "_"*right_side + "|", flush=True)
    time.sleep(0.25)


print("\nbefore: " + "\033[6;30;43m" + "This text uses 6;30;43 ." + "\033[0m" + " :end\n")


import time

def gold_bg(text):
    return "\033[6;30;43m" + text + "\033[0m"

# gold_bg test
for i in range(1,10):
    left_side = i
    right_side = (10-i)

    e = "E"
    # e = random.choice(emoji_list)

    print("|" + "_"*left_side + gold_bg(e) + "_"*right_side + "|", flush=True)
    time.sleep(0.25)

# by Zach

# REVERSE gold_bg test
for i in range(10,0,-1):
    left_side = i
    right_side = (10-i)

    e = "E"
    # e = random.choice(emoji_list)

    print("|" + gold_bg("_"*left_side) + e + gold_bg("_"*right_side) + "|", flush=True)
    time.sleep(0.25)


