# Here is a coding cell. Try it!

41 + 1


41+1


2**5 #welcome to python!



2+5


# This is a code cell.
# To run it, click play at left or type ctrl-enter (cmd-enter)


answer = input("What is your name? ")
print("Welcome,", answer)
print()

print("My favorite number is 42, naturally.")
fav_num = input("What is your favorite number? ")
print()

print("Nice!", fav_num, "is solid. And very numeric!")
print("Plus, it's close to the best number, 42.")





fav_num


print("fav_num is", fav_num)


# prompt: fav_num + 1

fav_num_int = int(fav_num)
fav_num_int + 1


# This is a code-cell, which should be runnable with the arrow,
# also with (control-enter) or (command-return)

35 + 7


# Try running this cell to compute a googol
# This is the number after which Google was named!

10**100   # ** is power operator


# This cell imports the math library and then runs a function from it.
# (The square root of 4 is 2, as you'll see :-)

from math import *    # imports the sqrt function and others
sqrt(4)


# Here is an example of the "factorial" function. factorial(4) is 1*2*3*4, which is 24

factorial(10)     # also imported from the math library


1*2*3*4*5*6*7*8*9*10


# This cell shows how printing works. Try it!
# It should print:   Zero is 0
#             and:   One is 1       actually, 1.0, This is ok!
#
# Notice that this sets you up for the Four fours challenge!

print("Zero is", 4+4-4-4)
print("One is", (4/4) * (4/4))                 # Uh oh! No non-fours allowed!!
print("Two is", (4/4) + (4/4)  )   # Uh oh! This is incorrect... temporarily






-4 + sqrt(4) + factorial(4) - 4


#
#  Here is a space for your explorations and solutions
#       to the Four fours challenge.
#
# This will get you started.

import math

print("Zero is", 4+4-4-4)
print("One is", (4/4)*(4/4))  # yay!  exactly four fours - and no other digits
print("Two is", (4/4)+(4/4))
print("Three is", (4 + 4 + 4) / 4)
print("Four is", 4*(4-4)+4)
print("Five is", (4 * 4 + 4) / 4)
print("Six is", (4 + 4) / 4 + 4)
print("Seven is", 4 + 4 - 4 / 4)
print("Eight is", 4 + 4 + 4 - 4)
print("Nine is", 4 + 4 + 4 / 4)
print("Ten is", 44 / 4.4)
print("Eleven is", 4/.4+4/4)
print("Twelve is", (44+4)/4)
print("Thirteen is", math.factorial(4)-44/4)
print("Fourteen is", 4*(4-.4) -.4)
print("Fifteen is", 44/4+4)
print("Sixteen is", 4+4+4+4)
print("Seventeen is", 4*4+4/4)
print("Eighteen is", 44 * .4 +.4)
print("Nineteen is", math.factorial(4) -4 -4/4)
print("Twenty is", 4 * (4/4 +4))
print("Twenty-one is", (4.4+4)/.4) # I finally did it.. it is done






# prompt: math.sqrt(4)

print(math.sqrt(4))



math.factorial(4)


# prompt: help me generate the ints from 1 to 25 with only four fours

print("Two is", (4/4) + (4/4))
print("Three is", (4 + 4 + 4) / 4)
print("Four is", 4 + 4 - 4 + 4)
print("Five is", (4 * 4 + 4) / 4)
print("Six is", (4 + 4) / 4 + 4)
print("Seven is", 4 + 4 - 4 / 4)
print("Eight is", 4 + 4 + 4 - 4)
print("Nine is", 4 + 4 + 4 / 4)
print("Ten is", 44 / 4.4)
print("Eleven is", 44 / 4 + 4/4)
print("Twelve is", 4*4 -4 -4)
print("Thirteen is", 4 * 4 - 4 / 4)
print("Fourteen is", 4 * 4 - 4+4/4)
print("Fifteen is", 4 * 4 - 4 + (4/4))
print("Sixteen is", 4+4+4+4)
print("Seventeen is", 4*4+4/4)
print("Eighteen is", 4*4+4+4/4)
print("Nineteen is", 4*4+4+4/4)
print("Twenty is", 4 * 4 + 4 + 4/4)
print("Twenty-one is", 4*4+4+4/4)
print("Twenty-two is", 4*4+4+4/4)
print("Twenty-three is", 4*4+4+4/4)
print("Twenty-four is", 4*4+4+4/4)
print("Twenty-five is", 4*4+4+4/4)


#
# examples of the most important four types of data
#   Numeric:     int, float
#   Sequential:  str, list
#

print("Our examples:")
print("My examples?")


fav_num = 27      # an int
pi = 3.14         # a float

inator_creator = 'Dr. String'  # a string
L = ["jan", 31, "feb", 28, "mar", 31]       # a list, of ints and strings

#
# now, let's print them -- and print their types:
#
print("fav_num is", fav_num, "\n  type(fav_num) is", type(fav_num))
print()  # prints a blank line, "\n" is another way to go to a newline

print("pi is", pi, "\n  type(pi) is", type(pi))
print()

print("inator_creator is", inator_creator, "\n  type(inator_creator) is", type(inator_creator))
print()

print("L is", L, "\n  type(L) is", type(L))
print()


#
# You'll notice that Python uses <class 'int'> instead of int, and so on...
# No worries!
#


#
# Here, create at least one example of int, float, str, and list data
#
#     + Feel free to copy-and-paste-then-edit from the above cell
#
#     + The objective is to try out variables, assignment, and printing
#     + with your own names, data, etc.
#
#     + Try it out...  Creativity is the goal...
#
#     + Errors?  Ask us - or whoever's next to you... :-)
#
#

print("My examples?")


fav_num = 27      # an int
pi = 3.14         # a float

inator_creator = 'Dr. String'  # a string
L = [ 1, "fish", 2, "Phish", "blue", "red"]       # a list, of ints and strings

#
# now, let's print them -- and print their types:
#
print("fav_num is", fav_num, "\n  type(fav_num) is", type(fav_num))
print()  # prints a blank line, "\n" is another way to go to a newline

print("pi is", pi, "\n  type(pi) is", type(pi))
print()

print("inator_creator is", inator_creator, "\n  type(inator_creator) is", type(inator_creator))
print()

print("L is", L, "\n  type(L) is", type(L))
print()




#
# example of clobbering fav_num
#

print("fav_num before is", fav_num)

fav_num = 1925            # Founded it!  (This was CGU's founding year.)

print("fav_num after is", fav_num)
print("It overwrote my favorite number when I re-ran the cell....")


#
# Thought experiment:
#   What do you think happens if you re-run the above cell?!
#   Try it out... No need to answer the question beyond trying it! :-)
#


comp = "rock"
user = "paper"

if comp == 'paper' and user == 'paper':
    print('We tie. Try again?')

elif comp == 'rock':

    if user == 'scissors':
        print('I win! *_*')
    else:
        print('You win. Aargh!')


# A very unfair game of rock-paper-scissors

user_name = input("What is your name? ")
print("Welcome,", user_name, "!")
print()

print("I challenge you to rock-paper-scissors! Prepare to be vanquished!")

user_choice = input("Which do you choose? (rock/paper/scissors):")
print()

if user_choice == "rock":
  print("You chose", user_choice)
  print("But I chose paper...")   # this doesn't seem fair...
  print("My paper smothers your rock!")

if user_choice == "paper":
  print("You chose", user_choice, "?")
  print("I'm sorry: My scissors have confetti'd you!")

else:
  print("You chose", user_choice)
  print("Wow. You chose poorly!")
  print("I win.")


print()
print("Try again, if you dare...")



#
# A fairer but incomplete game of rock-paper-scissors
#

import random       # imports a library (named random)
                    # for making random choices

user_name = input("What is your name? ")
print("Welcome,", user_name, "!")
print()

print("Ready for RPS? Choose wisely!!")
user_choice = input("Which do you choose? (rock/paper/scissors):")
comp_choice = random.choice(["rock","paper","scissors"])
print()

# We print both choices
print("You chose", user_choice)
print("  I chose", comp_choice)
print()


# We can handle different possibilities with nested conditionals

if user_choice == "rock":

  if comp_choice == "rock":
    print("Aargh! We tied (but my rock was rockier!)")
  elif comp_choice == "paper":
    print("I win! My paper outclasses any rock!")
  else:
    print('Alack! Your rock cast my scissors asunder.')

else:

  print("You have ventured into the unknown...")
  print("... but I probably won!!")



print()
print("Try again!")



#
# A fairer but incomplete game of rock-paper-scissors
#

import random       # imports a library (named random)
                    # for making random choices

user_name = input("What is your name? ")
print("Welcome,", user_name, "!")
print()

print("Ready for RPS? Choose wisely!!")
user_choice = input("Which do you choose? (rock/paper/scissors):")
comp_choice = random.choice(["rock","paper","scissors"])
print()

# We print both choices
print("You chose", user_choice)
print("  I chose", comp_choice)
print()


# We can handle different possibilities with nested conditionals

if user_choice == "rock":

  if comp_choice == "rock":
    print("Aargh! We tied (but my rock was rockier!)")
  elif comp_choice == "paper":
    print("I win! My paper outclasses any rock!")
  else:
    print('Alack! Your rock cast my scissors asunder.')

if user_choice == "paper":

  if comp_choice == "rock":
    print("Alack! Your paper covers my rock!")
  elif comp_choice == "paper":
    print("Aargh! We tied (but my paper was paperier!)")
  else:
    print('I win! My scissors cut your paper!')

if user_choice == "scissors":

  if comp_choice == "scissors":
    print("Aargh! We tied (but my scissors were scissorsier!)")
  elif comp_choice == "rock":
    print("I win! My rock smashes your scissors!")
  else:
    print("Alack! Your scissors cut my paper!")



print()
print("Try again!")


# Title for this example adventure:   The Quest.
#
# Notes on how to "win" or "lose" this adventure:
#   To win, choose the table.
#   To lose, choose the door.


def adventure():
    """This function runs one session of interactive fiction
       Well, it's "fiction," depending on the pill color chosen...
       Arguments: no arguments (prompted text doesn't count as an argument)
       Results: no results     (printing doesn't count as a result)
    """
    user_name = input("What do they call you, worthy adventurer? ")

    print()
    print("Welcome,", user_name, "-- to the UnderMine, a labyrinth")
    print("beneath CGU's unending academic complex, with weighty")
    print("wonders and unreal quantities...of poptarts!")
    print()

    print("Your quest: To find--and partake of--a poptart!")
    print()
    flavor = input("What flavor do you seek? ")
    if flavor == "strawberry":
        print("Wise! You show deep poptart experience.")
    elif flavor == "s'mores":
        print("The taste of the campfire: well chosen, adventurer!")
    else:
        print("Each to their own, then.")
    print()

    print("On to the quest!\n\n")
    print("A corridor stretches before you; its dim lighting betrays, to")
    print("one side, a table supporting nameless forms of inorganic bulk")
    print("and, to the other, a door ajar, leaking laughter--is that")
    print("laughter?--of lab-goers.")
    print()

    choice1 = input("Do you choose the table or the door? [table/door] ")
    print()

    if choice1 == "table":
        print("As you approach the table, its hazy burdens loom ever larger,")
        print("until...")
        print()
        print("...they resolve into unending stacks of poptarts, foil")
        print("shimmering.  You succeed, sumptuously, in sating the")
        print("challenge--and your hunger.")
        print("Go well,", user_name, "!")

    else:
        print("You push the door into a gathering of sagefowl, athenas,")
        print("and stags alike, all relishing their tasks. Teamwork and")
        print("merriment abound here, except...")
        print()
        print("...they have consumed ALL of the poptarts! Drifts of wrappers")
        print("coat the floor.  Dizzy, you grasp for a pastry. None is at")
        print("hand. You exhale and slip under the teeming tide of foil as")
        print("it finishes winding around you.")
        print("Farewell,", user_name, ".")


#
# The above adventure is defined in a _function_  (next week: functions!)
#
# This means it's defined, but it does not run until we call it.
# This line below calls it:
adventure()


# Title for this adventure:   Dark or Light
#



def adventure():
    user_name = input("What's your name?")

    print()
    print("Welcome,", user_name, "to a simple crossroads")
    print("where sunlight and darkness meet")
    print()

    print("Take a choice")
    print()
    brightness = input("sunlight or moonlight?")
    if brightness == "sunlight":
        print("The warmth of the sunlight will shine upon you")
    elif brightness == "moonlight":
        print("A night owl, huh? moonlight guides your way")
    else:
        print("Indecision will plague you for the rest of eternity")
    print()

    print("How will you proceed then. \n\n")
    print("You look forward as two paths lie before you")
    print("one side, A bristling town with a large singular tower looms")
    print("in the distance.")
    print("On the other side, a lush forest, with pine trees and a dirt path")
    print()

    choice1 = input("Do you choose the town or the forest? [town/forest] ")
    print()

    if choice1 == "town":
        print("As you walk towards the town, you start to hear the hustle")
        print("and bustle of a crowd of people and as you listen closer")
        print("they start sounding angry and chase you down")
        print("they tackle you to the floor and slash at your body with")
        print("kitchen knives and pitchforks")
        print("you hear them shouting 'Witch!'")
        print("You die,", user_name, "!")

    else:
        print("As you walk towards the forest, you start to hear howling")
        print("and as you continue, the howling gets closer and more feral")
        print("A pack of wolves jump from the trees and tear at your neck")
        print("You die", user_name, ".")


#
# The above adventure is defined in a _function_  (next week: functions!)
#
# This means it's defined, but it does not run until we call it.
# This line below calls it:
adventure()


