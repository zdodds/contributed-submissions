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
    this_guess = choice(range(0,100))  # 0 to 99, inclusive

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
# A. Recursive Version:
def spongebobbify_each_R(s):
    from random import choice


    def spongebobbify_char(ch):
        return choice([ch.upper(), ch.lower()])
    if s == "":
        return ""

    return spongebobbify_char(s[0]) + spongebobbify_each_R(s[1:])




# Try our tests (next cell) and add three of your own...
# Preferably sBoB QuoTeS...


# Tests to try for spongebobbify_once
# There are not "right" answers, so we just test them:

print(spongebobbify_each_R('F is for friends who do stuff together!'))
print(spongebobbify_each_R('I knew I shouldn\'t have gotten out of bed today.'))
print(spongebobbify_each_R('The inner machinations of my mind are an enigma. - Patrick'))
print()

# Your tests here:
print(spongebobbify_each_R("The best time to wear a striped sweater is all the time."))
print(spongebobbify_each_R("Stupidity isn’t a virus, but it sure is spreading like one."))
print(spongebobbify_each_R("Always follow your heart unless your heart is bad with directions."))


#
# Use this cell -- and/or create more cells -- for your encode and decode functions
# There will be four functions in total!
#
# Characters we want to encode:
encode_src = "aeioubstcg"

# Matching symbols to encode:
encode_tgt = "@3!0µß$+©9"


def encode_once(ch):
    # Find the index of ch in the source string
    idx = encode_src.find(ch)
    if idx != -1:
        # If found, return the matching target character
        return encode_tgt[idx]
    else:
        # Otherwise, return unchanged
        return ch

def decode_once(ch):
    idx = encode_tgt.find(ch)
    if idx != -1:
        return encode_src[idx]
    else:
        return ch

def encode_each(s):
    result = ""
    for ch in s:
        result += encode_once(ch)
    return result

def decode_each(s):
    result = ""
    for ch in s:
        result += decode_once(ch)
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
## Example test
original = "I have to get a coffee."
encoded  = encode_each(original)
decoded  = decode_each(encoded)

print("Original:", original)
print("Encoded :", encoded)
print("Decoded :", decoded)  # Should match "I have to get coffee."

# Two fun "treasure-hunt" test sentences:
test2 = """When the moon is full, the silent dunes will speak the verse of midnight starlight,
           and an iron door conceals the vault guarded by living flames will be revealed."""
test3 = """My name is Maximus Decimus Meridius, commander of the Armies of the North,
           General of the Felix Legions and loyal servant to the true emperor, Marcus Aurelius.
           Father to a murdered son, husband to a murdered wife. And I will have my vengeance,
           in this life or the next.” (Gladiator, 2000)."""

# Encode and decode each sentence
encoded2 = encode_each(test2)
decoded2 = decode_each(encoded2)

encoded3 = encode_each(test3)
decoded3 = decode_each(encoded3)

# Print results
print("Original 2", test2)
print("Encoded 2 :", encoded2)
print("Decoded 2 :", decoded2)
print()
print("Original 3:", test3)
print("Encoded 3 :", encoded3)
print("Decoded 3 :", decoded3)



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
  total = 0
  for ch in s:
        total += vwl_once(ch)
  return total

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

def vowel_density(s):
    # number of vowels
    v_count = vwl_count(s)
    # total characters
    length = len(s)
    if length == 0:
        return 0
    else:
        return v_count / length

CGU_density = vowel_density(CGU)
CMC_density = vowel_density(CMC)
SCR_density = vowel_density(SCR)

print("CGU density =", CGU_density)
print("CMC density =", CMC_density)
print("SCR density =", SCR_density)

densities = [
    ("CGU", CGU_density),
    ("CMC", CMC_density),
    ("SCR", SCR_density)
]
densities.sort(key=lambda x: x[1], reverse=True)  # sort descending by density

print("\nMission statement with highest vowels-per-character is:", densities[0][0])




#
# Part 2 task: determine whose prose is more vowel-rich?
# + find a paragraph of prose you've written
# + find a paragraph a friend has written (or another one that you have!)
#
# Assign each to a variable:

Hope = """  From the crumbling watchtower, the lone sentinel gazed upon the horizon,
where storm clouds gathered like an omen. The ancient kingdom’s banners lay torn
amidst the rubble, and whispers of a forgotten prophecy echoed through the wind.
Despite the looming darkness, a spark of hope burned in the hearts of those
who still remembered the old legends.
"""

Fight = """  High above the silent planet, a colossal flagship drifted through
the cosmic haze. Its corridors glowed with flickering lights, revealing the scars
of countless battles. Within its steel chambers, the crew prepared for a final stand,
uncertain whether they would safeguard their galaxy or witness its last dawn.
"""

print("Hope ratio :", vowel_density(Hope))
print("Fight ratio:", vowel_density(Fight))

if vowel_density(Hope) > vowel_density(Fight):
    print("Hope paragraph is more vowel-rich!")
elif vowel_density(Hope) < vowel_density(Fight):
    print("Fight paragraph is more vowel-rich!")
else:
    print("Both paragraphs have the same vowel ratio!")




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
M = list(range (1,15) )
print(M)



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
#
# we wouldn’t want to use range(...) if our numbers aren’t in a sequence
# or if we have only a few specific values to loop through.



# How could we get this to print "Happy birthday!" 42 times?
total = 0
for i in range(42):
    total= i + 1
    print('Happy birthday!')
    print("total = ", total)



# return _after_ a loop:

def funA():
    for i in range(0,3):
        print("i is", i)
    return

# why does this not print anything?!??
# The function will not run until it is called by its name.
# I call it here:
funA()



# return _within_ a loop:

def funB():
    for i in range(0,3):
        print("i is", i)
        return

# What do we need here?  (Is this what you expect?!)
# When a return statement is inside a loop, it stops the entire function immediately on the first iteration.
# It’s “stronger” than the loop.

funB()

# If we want to print the whole range of the list, we have to put the "return" outside the loop
def funB3():
    for i in range(0,3):
        print("i is", i)
    return

# Function Call:
funB3()




# let's add an if statement (a conditional)
#              ... to test different indentations

def funB1():
    for i in range(1,6):
        if i%2 == 0:
            print("i is", i)
            return
funB1()

# when the test is true "i == 2", the return statement executes and entire function ends immediately.
# it never gets to check i = 3, 4, or 5.



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




#Loops we tried in our "breakout quiz":

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

#Loops we tried in our "breakout quiz":

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
    total = 0
    for x in L:
        total = total + x
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
    for x in L:
        if x % 2 != 0:
            total = total + x
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
    for x in L:
        if x != exc:
            total = total + x
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
    for x in L:
        if x == exc:
            break
        total = total + x
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

def guess_between(low,high):
    """ guesses a # from 0 to 99 (inclusive) until
        it gets one that is strictly less than high and
        greater than or equal to low
        Then, this function returns the total # of guesses made
    """
    guess = -1
    number_of_guesses = 0


    while not (low <= guess < high):
        guess = choice(range(0, 100))
        number_of_guesses += 1

    return number_of_guesses



#
# be sure to test your guess_between here -- and leave the test in the notebook!
#
print("guess_between(40,50) ->", guess_between(40,50))
print("guess_between(10,20) ->", guess_between(10,20))
print("guess_between(20,80) ->", guess_between(20,80))
print("guess_between(21,42) ->", guess_between(21,42))



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

    L = []

    while True:
        g = choice(range(1, high+1))
        if g in L:
            L.append(g)
            return L
        else:
            L.append(g)


#
# be sure to test your listTilRepeat here -- and leave the test in the notebook!
#
print("listTilRepeat(10) ->", listTilRepeat(10))
print("listTilRepeat(10) ->", listTilRepeat(10))
print("listTilRepeat(18) ->", listTilRepeat(18))
print("listTilRepeat(42) ->", listTilRepeat(42))



# The birthday paradox is the fact that
#     listTilRepeat(365) has surprisingly few elements!
#
# Run listTilRepeat(365) a few times and print its _length_ each time
#     (Don't print the lists... it's too much to digest.)
#
# To many people, the results feel counterintuitive!

# print("listTilRepeat(365) ->", listTilRepeat(365))



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
import time # Import the time module

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
# rwalk(15,0,30)



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


### Simulation Game ###
#
#The Cats, Mouse and Cheese Game
#
# cat1 = "🐱"
# cat2 = "🐈"
# mouse= "🐭"
# cheese = "🧀"
#
''' The cheese appears in a random location, and the mouse attempts to catch
the cheese by moving randomly. Each cat randomly moves left or right with each
step. If a cat lands on the mouse, that cat scores a point, and the mouse
respawns in a different random position. The same process applies to the
mouse trying to catch the cheese.
'''

import random
import time

def cats_mouse_cheese_game():
    # Emojis
    cat1_emoji = "🐱"
    cat2_emoji = "🐈"
    mouse_emoji = "🐭"
    cheese_emoji = "🧀"

    world_length = 14  # This means we have 15 horizontal positions in each round.
    max_steps = 30

    # Initial positions
    cat1_pos = 2
    cat2_pos = 17
    mouse_pos = 10
    cheese_pos = random.randint(0, world_length)

    cat1_score = 0
    cat2_score = 0
    mouse_score = 0

    for step in range(1, max_steps + 1):
        cat1_pos += random.choice([-1, 1])
        cat2_pos += random.choice([-1, 1])
        mouse_pos += random.choice([-1, 1])

        cat1_pos = max(0, min(cat1_pos, world_length))
        cat2_pos = max(0, min(cat2_pos, world_length))
        mouse_pos = max(0, min(mouse_pos, world_length))

        caught_messages = []

        mouse_caught = False

        if cat1_pos == mouse_pos:
            cat1_score += 1
            caught_messages.append("Cat1 caught the mouse!")
            mouse_caught = True

        if cat2_pos == mouse_pos:
            cat2_score += 1
            caught_messages.append("Cat2 caught the mouse!")
            mouse_caught = True

        if mouse_caught:
            # Respawn the mouse at a new random position
            mouse_pos = random.randint(0, world_length)

        if mouse_pos == cheese_pos:
            mouse_score += 1
            caught_messages.append("Mouse caught the cheese!")

            cheese_pos = random.randint(0, world_length)

        line = ""
        for i in range(world_length + 1):

            if i == cat1_pos and i == cat2_pos and i == mouse_pos and i == cheese_pos:
                line += cat1_emoji + cat2_emoji + mouse_emoji + cheese_emoji
            elif i == cat1_pos and i == cat2_pos and i == mouse_pos:
                line += cat1_emoji + cat2_emoji + mouse_emoji
            elif i == cat1_pos and i == cat2_pos and i == cheese_pos:
                line += cat1_emoji + cat2_emoji + cheese_emoji
            elif i == cat1_pos and i == mouse_pos and i == cheese_pos:
                line += cat1_emoji + mouse_emoji + cheese_emoji
            elif i == cat2_pos and i == mouse_pos and i == cheese_pos:
                line += cat2_emoji + mouse_emoji + cheese_emoji
            elif i == cat1_pos and i == cat2_pos:
                line += cat1_emoji + cat2_emoji
            elif i == cat1_pos and i == mouse_pos:
                line += cat1_emoji + mouse_emoji
            elif i == cat1_pos and i == cheese_pos:
                line += cat1_emoji + cheese_emoji
            elif i == cat2_pos and i == mouse_pos:
                line += cat2_emoji + mouse_emoji
            elif i == cat2_pos and i == cheese_pos:
                line += cat2_emoji + cheese_emoji
            elif i == mouse_pos and i == cheese_pos:
                line += mouse_emoji + cheese_emoji
            elif i == cat1_pos:
                line += cat1_emoji
            elif i == cat2_pos:
                line += cat2_emoji
            elif i == mouse_pos:
                line += mouse_emoji
            elif i == cheese_pos:
                line += cheese_emoji
            else:
                line += "_"

        # Print the scene + the scores so far
        print(
            f"{line} | step {step}, "
            f"cat1: {cat1_score}, cat2: {cat2_score}, mouse: {mouse_score}",
            flush=True
        )

        # If anything was caught, print those messages
        for msg in caught_messages:
            print("   ->", msg)

        # Pause briefly to watch the "animation"
        time.sleep(0.5)

    # After the loop, print final results
    print("\nFinal scores:")
    print("Cat1 :", cat1_score)
    print("Cat2 :", cat2_score)
    print("Mouse:", mouse_score)

    # Determine the winner
    highest = max(cat1_score, cat2_score, mouse_score)
    winners = []
    if cat1_score == highest:
        winners.append("Cat1")
    if cat2_score == highest:
        winners.append("Cat2")
    if mouse_score == highest:
        winners.append("Mouse")

    if len(winners) == 1:
        print(f"{winners[0]} wins! 🏆")
    else:
        print("It's a tie between:", " and ".join(winners), "!")

cats_mouse_cheese_game()



