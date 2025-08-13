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


import random
def spongebobbify_all(s):
  result = random.choice( [s.upper(), s.lower()] )   # choose one at random
  return result
def i2I_all_loop(x):

  result = ''        # the result starts as the EMPTY string!

  for c in x:        # c is the name we are giving to EACH element in s (that is, each character in s)
    result = result + i2I_once(c)
  return result


# Tests to try for spongebobbify_once
# There are not "right" answers, so we just test them:

print(spongebobbify_all('F is for friends who do stuff together!'))
print(spongebobbify_all('I knew I shouldn\'t have gotten out of bed today.'))
print(spongebobbify_all('The inner machinations of my mind are an enigma. - Patrick'))
print(spongebobbify_all('HeLL0 W0rld!'))
print(spongebobbify_all('To be or no to be - IST341_Participant_8!'))
print(spongebobbify_all('How much wood would a woodchuck'), end=" ")
print(i2I_all_loop('chuck if he couldn\'t chuck any wood!'))

print()

# but we want to use it on single letters!
print(spongebobbify_all('a'))
print(spongebobbify_all('b'))
print(spongebobbify_all('c'))

# Your tests here:


substitutions = {
    'a': '@', 'b': '8', 'c': '1', 'd': ']', 'e': '3',
    'f': '#', 'g': '9', 'h': '4', 'i': '!', 'j': ';',
}

reverse_substitutions = {v: k for k, v in substitutions.items()}

def encode_once(s):
    """Encodes a string by replacing defined characters once."""
    return ''.join(substitutions.get(char, char) for char in s)

def decode_once(s):
    """Decodes a string by reversing the substitution."""
    decoded = []
    i = 0
    while i < len(s):
        if s[i:i+2] in reverse_substitutions:  # Check for multi-character substitutions
            decoded.append(reverse_substitutions[s[i:i+2]])
            i += 2
        else:
            decoded.append(reverse_substitutions.get(s[i], s[i]))
            i += 1
    return ''.join(decoded)

def encode_each(s):
    """Encodes each character in the string recursively."""
    if not s:
        return ''
    return encode_once(s[0]) + encode_each(s[1:])

def decode_each(s):
    """Decodes each character in the string recursively."""
    if not s:
        return ''
    return decode_once(s[0]) + decode_each(s[1:])

def encode_all(text):
    """Encodes an entire text string."""
    return encode_each(text)

def decode_all(text):
    """Decodes an entire text string."""
    return decode_each(text)



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

print()
SCR = """The mission of Scripps College is to educate women to
develop their intellects and talents through active participation
in a community of scholars, so that as graduates they may contribute
to society through public and private lives of leadership, service,
integrity, and creativity.."""

E = encode_each(SCR)
print("encode_all(SCR) is", E)

print()
D = decode_each(E)
print("decode_all(E) is", D)  # should be the original!




NG = """Northrop Grumman's mission is to be at the forefront of technology and
innovation, delivering superior capability in tandem with maximized cost
efficiencies. The company is committed to advancing global security and human
discovery in support of its customers’ missions around the world. Northrop
Grumman's teams are exploring burgeoning research areas and creating revolutionary
technology that will not only power the mission but also connect, advance and
protect the U.S. and its allies. The company solves the toughest problems in
space, aeronautics, defense and cyberspace to meet the ever evolving needs of
its customers worldwid"""

N = encode_each(NG)
print("encode_all(NG) is", N)
print()
G = decode_each(N)
print("decode_all(N) is", G)  # should be the original!


USA = """To deploy, fight, and win our Nation’s wars by providing ready, prompt,
and sustained land dominance by Army forces across the full spectrum of conflict
as part of the Joint Force"""
print()
A = encode_each(USA)
print("encode_all(USA) is", A)
print()
S = decode_each(A)
print("decode_all(A) is", S)  # should be the original!


USR = """The Army Reserve's mission is to provide quick access to trained, equipped,
and ready Soldiers and units, with the critical enabling capabilities needed to
compete globally and win across the full range of military operations. The Army
Reserve is a powerful and resilient force that is ready to deliver vital capabilities
around the world and here at home. The Army Reserve provides combat-ready units and
soldiers to the Army and the Joint Force across the full spectrum of conflict. The
United States Special Operations Command (USSOCOM) Army Reserve Element provides
MOS-qualified and deployable Army Reserve personnel to augment USSOCOM."""
print()
U = encode_each(USR)
print("encode_all(USR) is", U)
print()
R = decode_each(U)
print("decode_all(U) is", R)  # should be the original!


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
  result = 0
  for c in s:
    result = result + vwl_once(c)
  return result



# Tests and tests-to-write are in the next cells:


print("vwl_count('English usually has lots of vowels.') should be 10 <->", vwl_count('English usually has lots of vowels.'))
print("The CGU mission statement has this many vowels: (let's see!) <->", vwl_count(CGU))


# Part 1 task: determine which mission statement has the most vowels-per-character!
#        Hint: count the vowels, then use len to determine vowels-per-character!
#        Compare the three strings already defined above:   CGU,  CMC,  and SCR


def ratio_vowels(text):
  return vwl_count(text) / len(text)
cmc_ratio = ratio_vowels(CMC)
scr_ratio = ratio_vowels(SCR)
cgu_ratio = ratio_vowels(CGU)

print("The CGU mission statement has this many vowels per character: (let's see!) <->", cgu_ratio)
print("The CMC mission statement has this many vowels per character: (let's see!) <->", cmc_ratio)
print("The SCR mission statement has this many vowels per character: (let's see!) <->", scr_ratio)
ratios = [("CGU", cgu_ratio,),("CMC", cmc_ratio), ("SCR", scr_ratio)]
winner = max(ratios, key=lambda x: x[1])
print("And the Winner is...", winner[0])


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

THEIRS = """  <paste their prose here
again, ok to have multiple lines inside
triple-quoted strings>
"""
yours_ratio = ratio_vowels(YOURS)
theirs_ratio = ratio_vowels(THEIRS)

print("YOURS ratio", yours_ratio)
print("THEIRS ratio", theirs_ratio)

if yours_ratio>theirs_ratio:
  print("YOURS is more vowel-rich!")
elif theirs_ratio>yours_ratio:
  print("THEIRS is more vowel-rich!")
else:
  print("They are equally vowel-rich!")



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
    print("Happy birthday!")




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



# an factorial function

def fac(N):
    """ a factorial function, returns the factorial of the input N
    """
    result = 1

    for x in range(1,N+1):
        result = result*x

    return result

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




# Function to sum all numbers in the list L
def summer(L):
    total = 0
    for num in L:
        total += num
    return total

# Function to sum only the odd numbers in the list L
def summedOdds(L):
    total = 0
    for num in L:
        if num % 2 != 0:  # Check if the number is odd
            total += num
    return total

# Function to sum all numbers in L except the given "exc"
def summedExcept(exc, L):
    total = 0
    for num in L:
        if num != exc:  # Skip the number if it matches exc
            total += num
    return total

def summedUpto(exc, L):
    total = 0
    for num in L:
        if num == exc:
            break  # Stop summing once we reach exc
        total += num
    return total

# examples:
print(summer( [2,3,4,1] ),    "->  10")
print(summedOdds( [2,3,4,1] ),    "->   4")
print(summedExcept( 4, [2,3,4,1] ), "->   6")
print(summedUpto( 4, [2,3,4,1] ), "->   5")


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
    """ include a short description here!
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
    """ include a short description here!
    """
    total = 0
    for num in L:
        if num == exc:
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


import random

# Function to guess a number between low and high
def guess_between(low, high):
    while True:
        guess = random.randint(low, high - 1)  # Generate a number in the range [low, high)
        if low <= guess < high:  # Check if the number is within the range
            return guess

# Function to accumulate a list of values until one repeats
def listTilRepeat(high):
    seen = set()  # A set to keep track of numbers we have already seen
    numbers = []  # List to store the numbers
    while True:
        num = random.randint(0, high - 1)  # Generate a number between 0 and high-1
        if num in seen:  # Check if the number is a repeat
            numbers.append(num)  # Append the repeat number to the list
            return numbers
        seen.add(num)  # Add the number to the set
        numbers.append(num)  # Add the number to the list

print(guess_between(40, 50))  # Will print a number between 40 and 50 (inclusive of 40, exclusive of 50)

print(listTilRepeat(10))  # Will print a list with numbers from 0 to 9 until one repeats

# examples (don't forget the randomness will change things!)
#
#print(guess_between(40,50))
#
# listTilRepeat(10)      ->   [4, 7, 8, 3, 7]     (the final # must be a repeat)
# listTilRepeat(10)      ->   [2, 1, 9, 9]     (the final # must be a repeat)



# here, write guess_between
#

def guess_between(low,high):
    """ guesses a # from 0 to 99 (inclusive) until
        it gets one that is strictly less than high and
        greater than or equal to low
        Then, this function returns the total # of guesses made
    """
    count = 0
    while True:
        guess = random.randint(0, 99)
        count += 1
        if low <= guess < high:
            return count


#
# be sure to test your guess_between here -- and leave the test in the notebook!
#

total_guesses = guess_between(40, 50)
print(f"Total guesses made: {total_guesses}")



# Try out adding elements to Lists

L = [3,4]
print("Before: L is", L)

guess = 42
L = L + [guess]
print(" After: L is", L)


# here, write listTilRepeat
def listTilRepeat(high):
    """ this f'n accumulates random guesses into a list, L, until a repeat is found
        it then returns the list (the final element should be repeated, somewhere)
    """
    L = []
    while True:
        guess = random.randint(0, high - 1)
        if guess in L:
            L.append(guess)
            return L
        L.append(guess)


result = listTilRepeat(365)
print("Length of the list:", len(result))



# The birthday paradox is the fact that
#     listTilRepeat(365) has surprisingly few elements!
#
# Run listTilRepeat(365) a few times and print its _length_ each time
#     (Don't print the lists... it's too much to digest.)
#
# To many people, the results feel counterintuitive!


# Example while loop: the "guessing game"

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


#Example Monte Carlo simulation: rolling two dice and counting doubles

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
            print("|" + "_"*left_side + "S" + "_"*right_side + "|", flush=True)  # a start of our "ASCIImation"
            time.sleep(0.042)

    # the code can never get here!

# You will need to add the right-hand side, too
# Then, improve your sleepwalker!

# let's try it!  rwalk(tart, low, high)
rwalk(5,0,10)
rwalk(15,0,30)



#
# Task #2:  create a _two_ sleepwalker animation!
#           For an example _idea_ see the next cell...

import time
import random

def random_step():
    """Returns a random step of -1 or 1."""
    return random.choice([-1, 1])

def print_sailors(posA, posB):
    """Prints the dock with two sailors moving on it."""
    if posA < posB:
        pLeft = posA; cLeft = "\033[6;33;41m" + "S" + "\033[0m"  # Sailor A (Red)
        pRight = posB; cRight = "\033[6;36;43m" + "S" + "\033[0m"  # Sailor B (Blue)
    else:
        pLeft = posB; cLeft = "\033[6;36;43m" + "S" + "\033[0m"
        pRight = posA; cRight = "\033[6;33;41m" + "S" + "\033[0m"

    left_space = (pLeft - 0)
    middle_space = (pRight - pLeft)
    right_space = (30 - pRight)

    print("Dock|" + "_" * left_space + cLeft + "_" * middle_space + cRight + "_" * right_space + "|Water", flush=True)

def drunken_sailors_race(posA, posB):
    """
    Simulates two drunken sailors wandering randomly on a dock (0 to 30).
    They both take random steps until one of them falls into the water.
    """
    steps = 0  # Count the number of steps

    while 0 < posA < 30 and 0 < posB < 30:  # Ensure both sailors are on the dock
        print_sailors(posA, posB)  # Display their positions
        posA += random_step()  # Move Sailor A
        posB += random_step()  # Move Sailor B
        steps += 1  # Increment step count
        time.sleep(0.05)  # Small delay for visualization

    # Determine which sailor fell off first
    if posA <= 0 or posA >= 30:
        print("Sailor A fell into the water!")
    else:
        print("Sailor B fell into the water!")

    return steps

# Example test run
steps_taken = drunken_sailors_race(10, 20)
print("The race ended in", steps_taken, "steps!")




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
        pSM = pSM + rs()           # take a random step for the mint-chocolate poptart...
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


