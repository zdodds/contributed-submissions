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


i2I_all('aliiiiiiiiiiiiiiien')


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
    print("result is", result)    # uncomment this to see the process, in action!

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
import random

def spongebobbify_all(s):
  """ returns the input, randomly "upper-cased" or "lower-cased" """
  if s == '':       # EMPTY case!
    return ''
  else:
    return spongebobbify_once(s[0]) + spongebobbify_all(s[1:])  # FIRST and REST


# Try our tests (next cell) and add three of your own...
# Preferably sBoB QuoTeS...


# Tests to try for spongebobbify_once
# There are not "right" answers, so we just test them:

print(spongebobbify_all('F is for friends who do stuff together!'))
print(spongebobbify_all('I knew I shouldn\'t have gotten out of bed today.'))
print(spongebobbify_all('The inner machinations of my mind are an enigma. - Patrick'))
print(spongebobbify_all("And why are you acting so nervous? And why are you sweating so much and why do you look so hungry? and- You like Krabby Patties, don't you, Squidward?")) #Iconic
print(spongebobbify_all('Hello Operator, Get me the navy!.... O_O Squidward, the Robots have taken over the navy!!!')) #Squidward: THE ROBOTS HAVE TAKEN OVER THE WORLD.... OUR WORLD!!
print(spongebobbify_all('Smitty Warbenjargermanjensen, He was number #1!!!')) #He just wanted his hat back...
# Your tests here:


#
# Use this cell -- and/or create more cells -- for your encode and decode functions
# There will be four functions in total!
#
#SOMETIMES I AM MAD AND WANT TO YELL IN CAPS LOCK
#sometimes I'm too lazy to caps lock
#Does this count as ten or more letters??

def encode_each(s):
    return s.upper()

def encode_all(s):
  return encode_once(text)

def decode_each(s):
    return s.lower()

def decode_all(s):
  return decode_once(text)


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
question = "Does this count as over 10 letters, for the sake of the grade? I think the only other way to encode something is to define each individual letter"
E = encode_each(question)
print(E)

D = decode_each(E)
print(D)  # should be the original!

possible_solution = "ChatGPT says to define a map and decode using DECODING_MAP = {v: k for k, v in ENCODING_MAP.items()} but that's kind of confusing..."
E = encode_each(possible_solution)
print(E)

D = decode_each(E)
print(D)  # should be the original!

dilemma = "Some days I feel like I have a decent grasp on python on a basic level but the sky is the limit when it comes to creativity and I can't get passed hello world statements"
E = encode_each(dilemma)
print(E)

D = decode_each(E)
print(D)  # should be the original!


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
  vowels = 0
  for c in s: #I got help from chatgpt which used "char", but the google colab auto generated a c.. does c always stand for char..?
      vowels += vwl_once(c)
  return vowels
  # you need to write this one!
  # use the previous examples (especially the "each" examples) as a guide! :-)



# Tests and tests-to-write are in the next cells:


print("vwl_count('English usually has lots of vowels.') should be 10 <->", vwl_count('English usually has lots of vowels.'))
print("The CGU mission statement has this many vowels: (let's see!) <->", vwl_count(CGU))


#
# Part 1 task: determine which mission statement has the most vowels-per-character!
print(vwl_count(CGU))
print(vwl_count(CMC))
print(vwl_count(SCR))
#        Hint: count the vowels, then use len to determine vowels-per-character!
#        Compare the three strings already defined above:   CGU,  CMC,  and SCR





#
# Part 2 task: determine whose prose is more vowel-rich?
# + find a paragraph of prose you've written
# + find a paragraph a friend has written (or another one that you have!)
#
# Assign each to a variable:

YOURS =  """There once was a man from Peru;who dreamed he was eating his shoe. He woke with a fright in the middle of the night to find that his dream had come true."""
 #Gary read this in one of spongebob's dreams...
print(vwl_count(YOURS))

THEIRS = """  meow, meow, meow, meow, meow, meow, meow
""" #Sigh.. he has such a way with words... - Sandy
print(vwl_count(THEIRS))
#
# This analysis is similar to the mission statements...
#
print('Seems like Gary might be more well-versed in dreams')



# this imports the library named random

import random

# once it's imported, you are able to call random.choice(L) for any sequence L
# try it:


import random
# Try out random.choice -- several times!
result = random.choice( ['claremont', 'graduate', 'university'] )
print("result is", result)



# let's see a loop do this 10 times!
total = 0

for i in range(10):                # loop 10 times
    result = random.choice( ['claremont', 'graduate', 'university'] ) # choose
    print("result is", result)     # print
    if result == 'claremont':
      total = total + 1

      print("Total is", total)


random.choice( 'mudd' )


list(range (0,10))[-3:]


random.uniform(41, 43)


random.choice( '1,2,3,4')



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
L = list(range(27,100))    # try different values; try omitting/changing the 0
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


age = 41


age = age + 1


age += 1






# How could we get this to print "Happy birthday!" 42 times?

for i in range(0,42):
    print('Happy birthday!')



# return _after_ a loop:

def funA():
    for i in range(0,3):
        print("i is", i)
    return
# why does this not print anything?!??




# return _within_ a loop:

def funB():
    for i in range(0,3):
       print("i is", i)
       return

# What do we need here?  (Is this what you expect?!)




# let's add an if statement (a conditional)
#              ... to test different indentations

def funB1():
    for i in range(1,6):
        if i%2 == 0:
            print("i is", i)
            return




# an add-em-up function

def addup(N):
    """ adds from 1 through N (inclusive)
    """
    result = 0

    for x in range(1,N+1):
        result = result + x

    return result

# addup(4) should be 0+1+2+3+4 == 10
print(addup(4))



# an factorial function

def fac(N):
    """ a factorial function, returns the factorial of the input N
    """
    result = 1

    for x in range(1,N+1):
        result = result*x

    return result

# fac(4) should be 1*2*3*4 == 24
print(fac(4))



'Loops we tried in our "breakout quiz"'

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
    if S[i-1] == ' ':
        result += S[i]
print(result)




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
    total = 0  # Initialize the total sum to 0
    for num in L:  # Iterate through each element in the list
        total += num  # Add the current element to the total
    return total  # Return the final sum


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
        if num % 2 != 0:
            total += num
    return total



# Here are two tests -- be sure to try them!
print("summedOdds( [2,3,4,1] )  should be 4 <->", summedOdds( [2,3,4,1] ))
print("summedOdds( [35,3,4,100] )  should be 38 <->", summedOdds( [35,3,4,100] ))


#
# here, write the summedExcept function!
#

def summedExcept( exc, L ):
    """ Sums all elements in L that are not equal to exc, which must be defined in the first part of the function
    """
    total = 0
    for num in L:
        if num != exc:
            total += num
    return total


# Here are two tests -- be sure to try them!
print("summedExcept( 4, [2,3,4,1] )  should be 6 <->", summedExcept( 4, [2,3,4,1] ))
print("summedExcept( 4, [35,3,4,100] )  should be 138 <->", summedExcept( 4, [35,3,4,100] ))


#
# here, write the summedUpto function!
#

def summedUpto( exc, L ):
    """ Sums all elements in L until it reaches the value of exc, which must be defined in the first part of the function
    """
    total = 0
    for num in L:
        if num < exc:
            total += num
        else:
            break #break sounds like it just breaks all of the code..
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

# guess_between(low,high) #like guess, but until it gets a number anywhere between
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
import random
def guess_between(low,high):

    """ guesses a # from 0 to 99 (inclusive) until
        it gets one that is strictly less than high and
        greater than or equal to low
        Then, this function returns the total # of guesses made
    """
    guess_count = 0
    while True:
      guess = random.randint(0,99)
      guess_count += 1
      if low <= guess < high:
        return guess_count


#
# be sure to test your guess_between here -- and leave the test in the notebook!
#

# Example 1
print(guess_between(40, 50))  # Guesses until a number between 40 and 49 is found

# Example 2
print(guess_between(0, 10))  # Guesses until a number between 0 and 9 is found

# Example 3
print(guess_between(90, 100))  # Guesses until a number between 90 and 99 is found



# Try out adding elements to Lists

L = [3,4]
print("Before: L is", L)

guess = 42
L = L + [guess]
print(" After: L is", L)


#
# here, write listTilRepeat
#
import random

def listTilRepeat(high):
    """
    Accumulates random guesses into a list until a repeat is found.
    Returns the list, with the final element being a repeat of a previous element.
    """
    seen = set()  # To keep track of numbers already seen
    result = []   # To store the accumulated list
    while True:
        num = random.randint(0, high - 1)  # Generate a random number in [0, high)
        if num in seen:  # If the number has been seen before, stop
            result.append(num)  # Include the repeated number in the result
            return result
        seen.add(num)  # Add the number to the set of seen numbers
        result.append(num)  # Add the number to the result list

    """ this f'n accumulates random guesses into a list, L, until a repeat is found
        it then returns the list (the final element should be repeated, somewhere)
    """


#
# be sure to test your listTilRepeat here -- and leave the test in the notebook!
#
# Example 1
print(listTilRepeat(10))  # Guesses until a number repeats in the range [0, 9]

# Example 2
print(listTilRepeat(5))  # Guesses until a number repeats in the range [0, 4]

# Example 3
print(listTilRepeat(20))  # Guesses until a number repeats in the range [0, 19]




# The birthday paradox is the fact that
#     listTilRepeat(365) has surprisingly few elements!
#
# Run listTilRepeat(365) a few times and print its _length_ each time
#     (Don't print the lists... it's too much to digest.)
listTilRepeat(365)
print (len(listTilRepeat(365)))
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


import time
#
# Example Monte Carlo simulation: rolling two dice and counting doubles
#

from random import *

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


def sumN( N ):
  result = 0
  for x in range(1,N+1):
    print("x is", x)
    result += x
  return result
  sumN(4) #this didn't result anything, refer to lecture



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
            print("|" + "_"*left_side + "S" + "_"*right_side + "|", flush=True)  # a start of our "ASCIImation"
            time.sleep(0.042)

    # the code can never get here!

# You will need to add the right-hand side, too
# Then, improve your sleepwalker!

# let's try it!  rwalk(tart, low, high)
rwalk(5,0,10)
rwalk(15,0,30)



print(u"\U0001F415")
print(u"\U0001F436")


import time
import random

def dog():
    """One random step"""
    return random.choice([-1, 1])

def print_dog(bolt, chowder):
    """Prints the dogs' positions using emojis."""
    if bolt < chowder:
        dLeft_pos = bolt
        dRight_pos = chowder
        dLeft_emoji = "\U0001F415"  # Dog emoji
        dRight_emoji = "\U0001F436"  # Dog face emoji
    else:
        dLeft_pos = chowder
        dRight_pos = bolt
        dLeft_emoji = "\U0001F436"  # Dog face emoji
        dRight_emoji = "\U0001F415"  # Dog emoji

    left_space = dLeft_pos
    middle_space = dRight_pos - dLeft_pos - 1
    right_space = 30 - dRight_pos

    print("Out of Bounds|" + "_" * left_space + dLeft_emoji + "_" * middle_space + dRight_emoji + "_" * right_space + "|Out of Bounds", flush=True)

def dog_race(bolt, chowder):
    num_steps = 0
    while 0 < bolt < 30 and 0 < chowder < 30:
        print_dog(bolt, chowder)
        bolt += dog()
        chowder += dog()
        num_steps += 1
        time.sleep(0.05)
    return num_steps

# Example usage
dog_race(10, 20)


dog_race(10,20)



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


