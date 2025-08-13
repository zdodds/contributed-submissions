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
  """ returns the input, randomly "upper-cased" or "lower-cased" """
  result = ''        # the result starts as the EMPTY string!
  for c in s:        # c is the name we are giving to EACH element in s (that is, each character in s)
    result += spongebobbify_once(c)
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
print(spongebobbify_all('Basketball is fun!'))
print(spongebobbify_all('IST341_Participant_10 caesar makes good music!'))
print(spongebobbify_all('helloooooooo'))
print()


#
# Use this cell -- and/or create more cells -- for your encode and decode functions
# There will be four functions in total!
#
def encode_once(s):
  """encodes a single character by subbing it w/ another character """
  encode_subs = {'a':'!', 'b':'@', 'c':'#', 'd':'$', 'e':'%', 'f':'^', 'g':'&', 'h':'*', 'i':'(', 'j':')'}
  if s in encode_subs:
    return encode_subs[s]
  else:
    return s

def decode_once(s):
  """decodes a single character by subbing it w/ another character """
  decode_subs = {'!':'a', '@':'b', '#':'c', '$':'d', '%':'e', '^':'f', '&':'g', '*':'h', '(':'i', ')':'j'}
  if s in decode_subs:
    return decode_subs[s]
  return s

def encode_each(s):
  """encodes a string by encoding each character """
  result = ''
  for c in s:
    result += encode_once(c)
  return result

def decode_each(s):
  """decodes a string by decoding each character"""
  result = ''
  for c in s:
    result += decode_once(c)
  return result

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
E = encode_each('abcdefghij')
print("encode_each('abcdefghij') is", E)
D = decode_each(E)
print("decode_all(E) is", D)

E = encode_each('hi my name is IST341_Participant_7')
print("encode_each('hi my name is IST341_Participant_7') is", E)
D = decode_each(E)
print("decode_all(E) is", D)

E = encode_each('i like to play basketball')
print("encode_each('i like to play basketball') is", E)
D = decode_each(E)
print("decode_all(E) is", D)

#


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
  count = 0
  for c in s:
    count += vwl_once(c)
  return count


# Tests and tests-to-write are in the next cells:


CGU = """Claremont Graduate University prepares individuals to be leaders
for positive change in the world. Unique in its transdisciplinary approach,
the university is dedicated to the creation, dissemination, and application
of new knowledge and diverse perspectives through research, practice,
creative works, and community engagement.
"""

CMC = """Claremont McKenna College's mission is to educate its students
for thoughtful and productive lives and responsible leadership in
business, government, and the professions, and to support faculty
and student scholarship that contribute to intellectual vitality
and the understanding of public policy issues."""

SCR = """The mission of Scripps College is to educate women to
develop their intellects and talents through active participation
in a community of scholars, so that as graduates they may contribute
to society through public and private lives of leadership, service,
integrity, and creativity.."""



print("vwl_count('English usually has lots of vowels.') should be 10 <->", vwl_count('English usually has lots of vowels.'))
print("The CGU mission statement has this many vowels: (let's see!) <->", vwl_count(CGU))


#
# Part 1 task: determine which mission statement has the most vowels-per-character!
#
#        Hint: count the vowels, then use len to determine vowels-per-character!
#        Compare the three strings already defined above:   CGU,  CMC,  and SCR
CGU_ratio = vwl_count(CGU) / len(CGU)
CMC_ratio = vwl_count(CMC) / len(CMC)
SCR_ratio = vwl_count(SCR) / len(SCR)

print("The CGU mission statement has this many vowels-per-character: <->", CGU_ratio)
print("The CMC mission statement has this many vowels-per-character: <->", CMC_ratio)
print("The SCR mission statement has this many vowels-per-character: <->", SCR_ratio)

print(max(CGU_ratio, CMC_ratio, SCR_ratio))




#
# Part 2 task: determine whose prose is more vowel-rich?
# + find a paragraph of prose you've written
# + find a paragraph a friend has written (or another one that you have!)
#
# Assign each to a variable:

YOURS = """The Databases Final Project compiled all of the skills and tools we learned throughout the semester, which I enjoyed. This project has shown me how to apply our SQL and database knowledge to a real-world business application. The guidelines for the project were straightforward, but didn’t tell us what exactly to do step-by-step. Usually, class projects tell us exactly what we need to do, but for this project we really had to hone in on the tools we learned in class to guide us through the project. I learned how to create a database in a team setting. I also learned how important it is to have a solid foundation before actually creating our database using SQL. For example, it was extremely helpful writing out why we need the database (business understanding), important information needed in the database (data understanding), and a layout of what we want our database to look like (conceptual design). Having this written out makes the security requirements, logical design, physical design, and database implementation process more efficient as we went through our project. Overall, I really enjoyed the challenge of this project, and I feel that it definitely helps set me up for the business world.
"""

THEIRS = """The Marvel Universe is one of the most interconnected fictional worlds ever created, encompassing over 6,000 characters across comics, films, television, and other media. Its complex narrative structure, characterized by a web of alliances, rivalries, and collaborations, provides a unique opportunity to study network dynamics within a fictional framework. While it may initially appear as mere entertainment, the Marvel Universe functions as a dynamic social network where characters are the nodes, and their interactions form the edges. This intricate structure is not just a storytelling mechanism but also a reflection of real-world social systems. Analyzing this network enables researchers to uncover patterns of influence, hierarchy, and community, much like those observed in human societies.This study focuses on two pivotal research questions"""


#
# This analysis is similar to the mission statements...
#
YOURS_ratio = vwl_count(YOURS) / len(YOURS)
THEIRS_ratio = vwl_count(THEIRS) / len(THEIRS)


print("my paragraph has this many vowels-per-character: <->", YOURS_ratio)
print("their paragraph has this many vowels-per-character: <->", THEIRS_ratio)
print(max(YOURS_ratio, THEIRS_ratio))



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

for i in range(0,42):
    print('Happy birthday!')



# return _after_ a loop:

def funA():
    for i in range(0,3):
        print("i is", i)
    return

# why does this not print anything?!??
funA()



# return _within_ a loop:

def funB():
    for i in range(0,3):
       print("i is", i)
       return

# What do we need here?  (Is this what you expect?!)
funB()



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
    sum_count = 0
    for i in L:
        sum_count += i
    return sum_count



# Here are two tests -- be sure to try them!
print("summer( [2,3,4,1] )  should be 10 <->", summer( [2,3,4,1] ))
print("summer( [35,3,4,100] )  should be 142 <->", summer( [35,3,4,100] ))


#
# here, write the summedOdds function!
#

def summedOdds(L):
    """ uses a for loop to add and return all of the _odd_ elements in L
    """
    sum_count = 0
    for i in L:
        if i % 2 == 1:
            sum_count += i
    return sum_count


# Here are two tests -- be sure to try them!
print("summedOdds( [2,3,4,1] )  should be 4 <->", summedOdds( [2,3,4,1] ))
print("summedOdds( [35,3,4,100] )  should be 38 <->", summedOdds( [35,3,4,100] ))


#
# here, write the summedExcept function!
#

def summedExcept( exc, L ):
    """ returns the sum of all elements excluding exc
    """
    sum_count = 0
    for i in L:
        if i != exc:
            sum_count += i
    return sum_count



# Here are two tests -- be sure to try them!
print("summedExcept( 4, [2,3,4,1] )  should be 6 <->", summedExcept( 4, [2,3,4,1] ))
print("summedExcept( 4, [35,3,4,100] )  should be 138 <->", summedExcept( 4, [35,3,4,100] ))


#
# here, write the summedUpto function!
#

def summedUpto( exc, L ):
    """ returns the sum of all elements up to the position of exc in L
    """
    sum_count = 0
    for num in L:
        if num == exc:
            break
        sum_count += num
    return sum_count


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
from random import *

def guess_between(low,high):
    """ guesses a # from 0 to 99 (inclusive) until
        it gets one that is strictly less than high and
        greater than or equal to low
        Then, this function returns the total # of guesses made
    """
    guess = -1      # start with a wrong guess and don't count it as a guess
    number_of_guesses = 0   # start with no guesses made so far...

    while guess < low or guess >= high:
        guess = choice( range(0,100) )  # 0 to 99, inclusive
        number_of_guesses += 1

    return number_of_guesses


#
# be sure to test your guess_between here -- and leave the test in the notebook!
#
print("guess_between(10, 20) ->", guess_between(10, 20))
print("guess_between(30, 32) ->", guess_between(30, 32))
print("guess_between(76, 93) ->", guess_between(76, 93))




# Try out adding elements to Lists

L = [3,4]
print("Before: L is", L)

guess = 42
L = L + [guess]
print(" After: L is", L)


#
# here, write listTilRepeat
#
from random import choice

def listTilRepeat(high):
    """ this f'n accumulates random guesses into a list, L, until a repeat is found
        it then returns the list (the final element should be repeated, somewhere)
    """
    L = []
    guess = -1

    while True:
        guess = choice( range(0,high) )
        if guess in L:
            break
        L = L + [guess]

    return L


#
# be sure to test your listTilRepeat here -- and leave the test in the notebook!
#
print("listTilRepeat(5) ->", listTilRepeat(5))
print("listTilRepeat(10) ->", listTilRepeat(10))
print("listTilRepeat(20) ->", listTilRepeat(20))





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
            print("🪩" + "🌫️"*left_side + "🕺🏽" + "🌫️"*right_side + "🪩", flush=True)  # a start of our "ASCIImation"
            time.sleep(0.042)

    # the code can never get here!

# You will need to add the right-hand side, too
# Then, improve your sleepwalker!

# let's try it!  rwalk(tart, low, high)
rwalk(5,0,10)
#rwalk(15,0,30)



#
# Task #2:  create a _two_ sleepwalker animation!
#           For an example _idea_ see the next cell...





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


import time
import random

def rs():
    """ One random step """
    return random.choice([-1, 1])

def print_dancers(dancer1, dancer2, trophy):
    """ Print the two dancers and the trophy between them """
    if dancer1 < dancer2:
        pLeft = dancer1;   cLeft = "🕺🏽"
        pRight = dancer2;  cRight = "🕺"
    else:
        pLeft = dancer2;   cLeft = "🕺"
        pRight = dancer1;  cRight = "🕺🏽"

    item_pos = (pLeft + pRight) // 2  # Place item in the middle
    left_space = (pLeft-0)
    middle_space = (item_pos - pLeft)
    right_space = (30-item_pos)

    print("🪩" + "_"*left_space + cLeft + "_"*middle_space + "🏆" + "_"*(pRight - item_pos) + cRight + "_"*right_space + "🪩", flush=True)

def dance_comp(dancer1, dancer2):
    """
        This simulator observes two dancers
        wandering towards an trophy between them.
    """
    num_steps = 0       # count the number of steps
    trophy = (dancer1 + dancer2) // 2  # Item starts between them

    while dancer1 != trophy and dancer2 != trophy:
        print_dancers(dancer1, dancer2, trophy)  # Print the current poptart-configuration!
        if dancer1 < trophy:
            dancer1 = max(0, min(30, dancer1 + rs()))  # Move towards the item
        if dancer2 > trophy:
            dancer2 = max(0, min(30, dancer2 + rs()))  # Move towards the item
        num_steps += 1             # Add 1 to our count of steps
        time.sleep(0.05)           # Pause a bit, to add drama!
    # winner
    winner = "🕺🏽" if dancer1 == trophy else "🕺"
    print(f"{winner} wins the dance competition!")

    # Finished with the while loop!
    return num_steps

# Example usage
print("Steps taken:", dance_comp(10, 15))



