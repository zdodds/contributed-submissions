





SO CUUUUTE


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
def spongebobbify_all(s):
  """ changes all characters to either uppercase or lowercase, randomly """
  result = ''
  for c in s:
    result = result + spongebobbify_once(c)
  return result



# Try our tests (next cell) and add three of your own...
# Preferably sBoB QuoTeS...


# Tests to try for spongebobbify_once
# There are not "right" answers, so we just test them:

print(spongebobbify_all('F is for friends who do stuff together!'))
print(spongebobbify_all('I knew I shouldn\'t have gotten out of bed today.'))
print(spongebobbify_all('The inner machinations of my mind are an enigma. - Patrick'))
print()

# Your tests here:
print(spongebobbify_all('I don\'t know any spongebob quotes help'))
print(spongebobbify_all('I only vaguely remember the song'))
print(spongebobbify_all('WHO LIVES IN A PINEAPPLE UNDER THE SEA?'))
print(spongebobbify_all('"SPONGEBOB SQUAREPANTS!"'))
print(spongebobbify_all('ABSORBENT AND YELLOW AND POROUS IS HE'))
print(spongebobbify_all('That\'s about it :\')'))


def encode_once(c):
  """ encodes a character in the first half of the alphabet
      (and some others), once """
  if c == ',': # , -> '
    return "'"
  elif c == "'": # ' -> .
    return '.'
  elif c == '.': # . -> ,
    return ','
  elif c == 'a': # a -> @ *rvrs
    return '@'
  elif c == '@': # @ -> a *rvrs
    return 'a'
  elif c == 'A': # A -> E *rvrs
    return 'E'
  elif c == 'b': # b -> d *rvrs
    return 'd'
  elif c == 'B':
    return 'D'
  elif c == 'c': # c <-> u *rvrs
    return 'u'
  elif c == 'C':
    return 'U'
  elif c == 'u':
    return 'c'
  elif c == 'U':
    return 'C'
  elif c == 'd': # d -> b * rvrs
    return 'b'
  elif c == 'D':
    return 'B'
  elif c == 'e': # e -> 3 * rvrs
    return '3'
  elif c == '3': # 3 -> E *rvrs
    return 'e'
  elif c == 'E': # E -> A * rvrs
    return 'A'
  elif c == 'f': # f -> l
    return 'l'
  elif c == 'F':
    return 'L'
  elif c == 'g': # g -> j *rvrs
    return 'j'
  elif c == 'G':
    return 'J'
  elif c == 'h': # h <-> p *rvrs
    return 'p'
  elif c == 'H':
    return 'P'
  elif c == 'p':
    return 'h'
  elif c == 'P':
    return 'H'
  elif c == 'i': # i <-> ! *rvrs
    return '!'
  elif c == '!':
    return 'i'
  elif c == 'I': # I <-> 1 *rvrs
    return '1'
  elif c == '1':
    return 'I'
  elif c == 'j': # j -> g *rvrs
    return 'g'
  elif c == 'J':
    return 'G'
  elif c == 'k': # k <-> w *rvrs
    return 'w'
  elif c == 'K':
    return 'W'
  elif c == 'w':
    return 'k'
  elif c == 'W':
    return 'K'
  elif c == 'l': # l -> f *rvrs
    return 'f'
  elif c == 'L':
    return 'F'
  else:
    return c


def decode_once(c):
  """ decodes a character in the first half of the alphabet
      (and some others), once """
  if c == "'": # , <- '
    return ","
  elif c == ".": # ' <- .
    return "'"
  elif c == ',': # . <- ,
    return '.'
  elif c == 'a': # a -> @ *rvrs
    return '@'
  elif c == '@': # @ -> a *rvrs
    return 'a'
  elif c == 'A': # A -> E *rvrs
    return 'E'
  elif c == 'b': # b -> d *rvrs
    return 'd'
  elif c == 'B':
    return 'D'
  elif c == 'c': # c <-> u *rvrs
    return 'u'
  elif c == 'C':
    return 'U'
  elif c == 'u':
    return 'c'
  elif c == 'U':
    return 'C'
  elif c == 'd': # d -> b * rvrs
    return 'b'
  elif c == 'D':
    return 'B'
  elif c == 'e': # e -> 3 * rvrs
    return '3'
  elif c == '3': # 3 -> E *rvrs
    return 'e'
  elif c == 'E': # E -> A * rvrs
    return 'A'
  elif c == 'f': # f -> l
    return 'l'
  elif c == 'F':
    return 'L'
  elif c == 'g': # g -> j *rvrs
    return 'j'
  elif c == 'G':
    return 'J'
  elif c == 'h': # h <-> p *rvrs
    return 'p'
  elif c == 'H':
    return 'P'
  elif c == 'p':
    return 'h'
  elif c == 'P':
    return 'H'
  elif c == 'i': # i <-> ! *rvrs
    return '!'
  elif c == '!':
    return 'i'
  elif c == 'I': # I <-> 1 *rvrs
    return '1'
  elif c == '1':
    return 'I'
  elif c == 'j': # j -> g *rvrs
    return 'g'
  elif c == 'J':
    return 'G'
  elif c == 'k': # k <-> w *rvrs
    return 'w'
  elif c == 'K':
    return 'W'
  elif c == 'w':
    return 'k'
  elif c == 'W':
    return 'K'
  elif c == 'l': # l -> f *rvrs
    return 'f'
  elif c == 'L':
    return 'F'
  else:
    return c


#
# Use this cell -- and/or create more cells -- for your encode and decode functions
# There will be four functions in total!
#
def encode_each(E):
  """encode all the characters!!!"""
  result = ''
  for c in E:
    result = result + encode_once(c)
  return result

def decode_each(D):
  """decode all the characters!!!"""
  result = ''
  for c in D:
    result = result + decode_once(c)
  return result




# Our tests are below. Then, add three tests of your own:


CGU = """Claremont Graduate University prepares individuals to be leaders
for positive change in the world. Unique in its transdisciplinary approach,
the university is dedicated to the creation, dissemination, and application
of new knowledge and diverse perspectives through research, practice,
creative works, and community engagement."""

E = encode_each(CGU)
print("encode_all(CGU) is '", E, "'", sep='')
print()

D = decode_each(E)
print("decode_all(E) is '", D, "'", sep='')   # should be the original!
print()

CMC = """Claremont McKenna College's mission is to educate its students
for thoughtful and productive lives and responsible leadership in
business, government, and the professions, and to support faculty
and student scholarship that contribute to intellectual vitality
and the understanding of public policy issues."""

E = encode_each(CMC)
print("encode_all(CMC) is '", E, "'", sep='')
print()
D = decode_each(E)
print("decode_all(E) is '", D, "'", sep='')   # should be the original!
print()

SCR = """The mission of Scripps College is to educate women to
develop their intellects and talents through active participation
in a community of scholars, so that as graduates they may contribute
to society through public and private lives of leadership, service,
integrity, and creativity.."""

E = encode_each(SCR)
print("encode_all(SCR) is '", E, "'", sep='')
print()
D = decode_each(E)
print("decode_all(E) is '", D, "'", sep='')  # should be the original!

# that is illegible :')


#
# Above - or here - include three encode/decode tests of your own...
#

ELR = """El Roble Intermediate School provides a warm, stimulating environment
 where students are actively involved in learning academics as well as positive
 values... We have made a commitment to provide the best educational
 program possible for El Roble Intermediate School's students... Together, through
 our hard work, each student is equipped academically, prepared socially, and
 empowered to participate and thrive in a a challenging, diverse world. """
E = encode_each(ELR)
print("encode_all(ELR) is '", E, "'", sep='')
print()
D = decode_each(E)
print("decode_all(E) is '", D, "'", sep='')
print()

CHS = """CHS is an extraordinarily special school that exemplifies more than a
century of academic excellence and tradition. Visitors to CHS frequently note
 he park and college-like atmosphere of our campus, and our student-centered
 approach to teaching and learning. As a truly comprehensive high school, CHS
 features a full range of co-curricular and extra-curricular activities... CHS
 offers Advanced Placement (AP), IB, AVID, and dual-enrollment Citrus College/CHS
 courses on our campus. CHS students who have exhausted our curricular options
 also have opportunities to take courses with the Claremont Colleges. CHS has
 been recognized by the California Department of Education as an Exemplary Arts
 Education Program..."""
E = encode_each(CHS)
print("encode_all(CGU) is '", E, "'", sep='')
print()
D = decode_each(E)
print("decode_all(E) is '", D, "'", sep='')
print()

WA = """ The wandering albatross, also known as the snowy albatross or the white
 winged albatross is a seabird in the Diomedeidae. Young wandering albatrosses
 are brown, and become whiter throughout their lifetime. They have the largest
 wingspan of any extant bird. By catching the wind with these large wings, they
 are able to stay in the air for long periods of time without flapping their wings.
 Most albatrosses never touch land for several years, instead landing on the water
 to rest. They can live for over 50 years."""
E = encode_each(WA)
print("encode_all(WA) is '", E, "'", sep='')
print()
D = decode_each(E)
print("decode_all(E) is '", D, "'", sep='')
print()


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



# Tests and tests-to-write are in the next cells:


print("vwl_count('English usually has lots of vowels.') should be 10 <->", vwl_count('English usually has lots of vowels.'))
print("The CGU mission statement has this many vowels: (let's see!) <->", vwl_count(CGU))


#
# Part 1 task: determine which mission statement has the most vowels-per-character!
#
#        Hint: count the vowels, then use len to determine vowels-per-character!
#        Compare the three strings already defined above:   CGU,  CMC,  and SCR
U = vwl_count(CGU)/len(CGU)
print('CGU has', U, 'vowels per character.')
C = vwl_count(CMC)/len(CMC)
print('CMC has', C, 'vowels per character.')
R = vwl_count(SCR)/len(SCR)
print('SCR has', R, 'vowels per character.')
print('CGU has the most vowels per character')


#
# Part 2 task: determine whose prose is more vowel-rich?
# + find a paragraph of prose you've written
# + find a paragraph a friend has written (or another one that you have!)
#
# Assign each to a variable:



# A thing we did for English :D

YOURS = """It swayed, calling on its friends—were they friends? Did it even have
 friends?— to help. The dim moonlight was blocked completely, plunging the forest
 floor into pitch blackness. It heard them begin to panic. Clearly they hadn’t
 had the foresight to bring a flashlight. It was pleased with itself, sending out
 waves of praise to its friends. Yes, they were the closest anything would come
 to be its friends."""

THEIRS = """There is something… off about these woods. I don’t know exactly what
 it was, it just felt weird. You know that feeling when you think someone is
 watching you? When you see odd shadows in the corners of your eyes, when you can
 hear your heart pounding in your ears, and the sense of dread. Dread. That’s what
 I’m feeling, deep in these woods. Like something horrible is just out of sight,
 toying with me."""

#
# This analysis is similar to the mission statements...
#
M = vwl_count(YOURS)/len(YOURS)
print('YOURS has', M, 'vowels per character.')
H = vwl_count(THEIRS)/len(THEIRS)
print('THEIRS has', H, 'vowels per character.')
print('THEIRS has the most vowels per character')

# :(
# I  n e e d  m o r e  v o w e l s . . .



# this imports the library named random

import random

# once it's imported, you are able to call random.choice(L) for any sequence L
# try it:
L = ['Hi', 'Hello', 'Morning', 'Greetings']
random.choice(L)



# Try out random.choice -- several times!
result = random.choice( ['claremont', 'graduate', 'university'] )
print("result is", result)

threes = random.choice([333, 33, 3, 33333, 999])
print()
print('threes is', threes)



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
L = list(range(300,333))    # try different values; try omitting/changing the 0
print(L)



# combining these, we can choose random integers from a list
result = choice( range(0,100) )   # from 0 to 99
print("result is", result)



# let's run this 10 times!
for i in range(0,10):
    result = choice( range(0,100) )   # from 0 to 99
    print("result is", result)



# let's get more comfortable with loops...

for i in [0,1,2]:     # Key: What variable is being defined and set?! i
    print("i is", i)




# Note that range(0,3) generates [0,1,2]

for i in range(0,3):     # Key: What variable is being defined and set?! i, again
    print("i is", i)

# When would you _not_ want to use range for integers? when they are not in a row



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
funA()
# now it does :D




# return _within_ a loop:

def funB():
    for i in range(0,3):
       print("i is", i)
       return

# What do we need here?  (Is this what you expect?!)




# return _within_ a loop:

def funB():
    for i in range(0,3):
       print("i is", i)
       return

# What do we need here?  (Is this what you expect?!)
# return just ends the whole function before the loop can happen :(



# let's add an if statement (a conditional)
#              ... to test different indentations

def funB1():
    for i in range(1,6):
        if i%2 == 0:
            print("i is", i)
            return

funB1()



# an add-em-up function

def addup(N):
    """ adds from 1 through N (inclusive)
    """
    result = 0

    for x in range(1,N+1):
        result = result + x

    return result

# addup(4) should be 0+1+2+3+4 == 10
addup(4)



# an factorial function

def fac(N):
    """ a factorial function, returns the factorial of the input N
    """
    result = 1

    for x in range(1,N+1):
        result = result*x

    return result

# fac(4) should be 1*2*3*4 == 24
fac(4)


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
#                                 exc is the "exception"
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
    result = 0
    for x in L:
      result += x
    return result


# Here are two tests -- be sure to try them!
print("summer( [2,3,4,1] )  should be 10 <->", summer( [2,3,4,1] ))
print("summer( [35,3,4,100] )  should be 142 <->", summer( [35,3,4,100] ))


#
# here, write the summedOdds function!
#

def summedOdds(L):
    """ uses a for loop to add and return all of the _odd_ elements in L
    """
    result = 0
    for x in L:
      if x%2 == 1:
        result += x
    return result



# Here are two tests -- be sure to try them!
print("summedOdds( [2,3,4,1] )  should be 4 <->", summedOdds( [2,3,4,1] ))
print("summedOdds( [35,3,4,100] )  should be 38 <->", summedOdds( [35,3,4,100] ))


#
# here, write the summedExcept function!
#

def summedExcept( exc, L ):
    """ return the sum of the numbers in L not equal to exc
    """
    result = 0
    for x in L:
      if x != exc:
        result += x
    return result



# Here are two tests -- be sure to try them!
print("summedExcept( 4, [2,3,4,1] )  should be 6 <->", summedExcept( 4, [2,3,4,1] ))
print("summedExcept( 4, [35,3,4,100] )  should be 138 <->", summedExcept( 4, [35,3,4,100] ))


#
# here, write the summedUpto function!
#

def summedUpto( exc, L ):
    """ return the sum of the numbers in L up to but not including exc
    """
    result = 0
    for x in L:
      if x == exc:
        return result
      result += x
    return result





# Here are two tests -- be sure to try them!
print("summedUpto( 4, [2,3,4,1] )  should be 5 <->", summedUpto( 4, [2,3,4,1] ))
print("summedUpto( 100, [35,3,4,100] )  should be 42 <->", summedUpto( 100, [35,3,4,100] ))



#
# don't count repeats... EHEHEHEHE
#

from random import *

def guess( hidden ):
    """
        have the computer guess numbers until it gets the "hidden" value
        return the number of guesses
    """
    guess = -1
    number_of_guesses = 0
    N = []

    while guess != hidden:
        guess = choice( range(0,100) )
        if guess in N:
          number_of_guesses += 0
          # print('repeat', guess) just testing it
        else:
          number_of_guesses += 1
          N += [guess]
          # print(guess) again :)

    return number_of_guesses

# test our function!
guess(42)



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

from random import *
def guess_between(low,high):
    """ guesses a # from 0 to 99 (inclusive) until
        it gets one that is strictly less than high and
        greater than or equal to low
        Then, this function returns the total # of guesses made
    """
    guess = -1
    guesses = 0
    N = []
    while guess not in range(low,high):
      guess = choice(range (0, 100))
      if guesses > 99:
        return 'not possible'
      else:
        if guess in N:
            guesses += 0
        else:
          guesses += 1
          N += [guess]
    return guesses



#
# be sure to test your guess_between here -- and leave the test in the notebook!
#
print(guess_between(10,50))

print(guess_between(80,90))

print(guess_between(1,2))

print(guess_between(900, 1000))



# Try out adding elements to Lists

L = [3,4]
print("Before: L is", L)

guess = 42
L = L + [guess]
print(" After: L is", L)


#
# here, write listTilRepeat
#

# .......

def listTilRepeat(high):
    """ this f'n accumulates random guesses into a list, L, until a repeat is found
        it then returns the list (the final element should be repeated, somewhere)
    """
    L = []
    guess = -1
    while guess not in L:
      L += [guess]
      guess = choice(range(high))
    L += [guess]
    return L


#
# be sure to test your listTilRepeat here -- and leave the test in the notebook!
#
print(listTilRepeat(40))
print(listTilRepeat(90))




# The birthday paradox is the fact that
#     listTilRepeat(365) has surprisingly few elements!
#
# Run listTilRepeat(365) a few times and print its _length_ each time
#     (Don't print the lists... it's too much to digest.)
#
# To many people, the results feel counterintuitive!
print(len(listTilRepeat(365)))
print(len(listTilRepeat(365)))
print(len(listTilRepeat(365)))



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
        d1 = choice( [1,2,3,4,5,6] )  # 1 to 6, inclusive
        d2 = choice( range(1,7) )     # 1 to 6, inclusive
        if d1 == d2:
            numdoubles += 1
            you = "🙂"
        else:
            you = " "

        print("run", i, "roll:", d1, d2, you, flush=True)
        time.sleep(.2)

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
        time.sleep(.1)

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
    """ One random step """
    return random.choice([-1, 1])

def rwalk(start, low, high):
    """ Random walk between -radius and +radius  (starting at 0 by default) """
    totalsteps = 0          # Initial value of totalsteps (_not_ final value!)
    current = start         # Current location (_not_ the total # of steps)

    while True:             # Run "forever" (really, until a return or break)
        if current <= low:  # too low!
            return totalsteps # Phew!  Return totalsteps (stops the while loop)
        elif current >= high: # too high!
            return totalsteps # Phew!  Return totalsteps (stops the while loop)
        else:
            current = current + rs()  # take a step
            totalsteps += 1           # count it

            # let's animate!
            left_side = current - low   # distance on the left of our sleepwalker
            right_side = high - current
            print("🛏️" + "ࡇ"*left_side + "(ᴗ˳ᴗ)" + "ࡇ"*right_side + "🎢", flush=True)  # a start of our "ASCIImation"
            time.sleep(0.21)

    # the code can never get here!

# You will need to add the right-hand side, too
# Then, improve your sleepwalker!

# let's try it!  rwalk(start, low, high)
rwalk(10,0,21)




#
# Task #2:  create a _two_ sleepwalker animation!
#           For an example _idea_ see the next cell...


# was not able to make the hole always like up because the birbs are not the same width as the emojis or the text :(

import time
import random

def rs():
    """ One random step """
    return random.choice([-1, 1])

def birbnest(stick):
  """ stick is in the middle and determines the lane sizes
      the birbs have to wander from the nest to the stick
      then back to the nest
      the birb that does that first wins
      they also have to try not to fall into the hole
      that gets revealed when both birds get the stick
  """
  end = stick*2
  current1 = 0
  current2 = (stick*2)
  gotstick = 0
  steps = 0
  birbone = '\033[6;37;41m' + '𓅛' + '\033[0m'
  birbtwo = '\033[6;37;46m' + '𓅪' + '\033[0m'

  while True:
    extras = ''
    # print(current1, current2, gotstick) # debugging
    if current1 <= 0 and (gotstick == 1 or gotstick == 3):
        print('birb one wins at step'  , steps)
        return
    elif current2 >= end and (gotstick == 2 or gotstick == 3):
        print('birb two wins at step ' , steps)
        return
    else:
      while current1 >= current2 or current1 == stick or current2 == stick:
        if current1 == stick == current2:
          if gotstick == 3:
            print('the two birbs collide and fall into the hole :(')
            return
          elif gotstick == 1:
            extras += ' birb one stops birb two from getting the stick. '
          elif gotstick == 2:
            extras += ' birb two stops birb one from getting the stick. '
          else:
            extras += ' the two birbs stops each other from getting the stick. '
        if current1 == stick:
          if gotstick == 0 or gotstick == 2:
            extras += ' birb one got the stick! '
            gotstick += 1
            if gotstick == 3:
              extras += ' it unveiled a hole! '
            current1 = stick - 1
          elif gotstick == 1:
            current1 = stick - 1
          else:
            extras += ' hop! '
            current1 += step1

        if current2 == stick:
          if gotstick == 1 or gotstick == 0:
            extras += ' birb two got the stick! '
            gotstick += 2
            if gotstick == 3:
              extras += 'it unveiled a hole!'
            current2 = stick + 1
          elif gotstick == 2:
            current2 = stick + 1
          else:
            extras += ' hop! '
            current2 += step2

        if current2 <= current1:
          extras += ' Oh wow! The two birbs say hi to each other! '
          current1 -= step1
          current2 -= step2

        if current1 >= end:
          current1 = end - 1

        if current2 <= 0:
          current2 = 1
      # print(current1, current2, gotstick) # more degubbing



      #animation time :') 𓅛 '
                        # 𓅪 '


      sym = []
      symbol = []

      L = [stick, current1, current2]
      for x in range(3):
        sym += [min(L)]
        if min(L) == stick:
          if gotstick != 3:
            symbol += ['🪵']
          else:
            symbol += ['🕳️']
        elif min(L) == current1:
          symbol += [birbone]
        elif min(L) == current2:
          symbol += [birbtwo]
        L.remove(min(L))
      # print(symbol)

      # print(sym)
      if sym[0] < stick:
        left = sym[0]
        mid1 = (sym[1] - sym[0]) - 1
        if sym[1] < stick:
          right = stick
          mid2 = (sym[2] - sym[1])-1
        else:
          right = end - sym[2]
          mid2 = (sym[2] - sym[1]) - 1
      elif sym[0] == stick:
        left = stick
        mid1 = (sym[1] - sym[0]) - 1
        mid2 = (sym[2] - sym[1]) - 1
        right = end - sym[2]




      # left = pos1
      # mid1 = pos2 - pos1
      # mid2 = pos3 - pos2
      # right = end - pos3

      # results = '🪺'+'.'*left + sym1 + '.'*mid1 + sym2 + '.'*mid2 + sym3 + '.'*right + '🪹'
      results = '🪺'+'_'*left + symbol[0] + '_'*mid1 + symbol[1] + '_'*mid2 + symbol[2] + '_'*right + '🪹'
      print(results, extras, flush=True)
      time.sleep(0.1)

      step1 = rs()
      current1 = current1 + step1
      step2 = rs()
      current2 = current2 + step2
      if current1 < 0:
        current1 = 0
      if current2 > end:
        current2 = end
      steps = steps + 1
      if steps > 200:
        print('This is taking too long. Let\'s just call it a tie.')
        return


birbnest(8)



birbnest(10)



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
        pSM = pSM + rs()
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
    return "\033[1;9;33m" + text + "\033[0m"

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


