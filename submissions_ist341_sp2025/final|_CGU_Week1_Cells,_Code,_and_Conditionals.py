# Here is a coding cell. Try it!

41 + 1


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


print("fav_num is", fav_num)


print(int(fav_num)+1)


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
print("One is", 1)                 # Uh oh! No non-fours allowed!!
print("Two is", (4/4) * (4/4)  )   # Uh oh! This is incorrect... temporarily






-4 + sqrt(4) + factorial(4) - 4


#
#  Here is a space for your explorations and solutions
#       to the Four fours challenge.
#
# This will get you started.

import math

#print("Zero is", 4+4-4-4)
#print("One is", (4/4)*(4/4))  # yay!  exactly four fours - and no other digits

print("Zero is", -4+4-4+4)
print("One is",(4*4)/(4*4))
print("Two is",(4/4)+(4/4))
print("Three is",(4+4+4)/4)
print("Four is",(4+4)-sqrt(4)-sqrt(4))
print("Five is",((44-math.factorial(4))/4))
print("Six is",((4 + 4) / 4) + 4)
print("Seven is",4+4-(4/4))
print("Eight is",4-4+4+4)
print("Nine is",4+4+(4/4))
print("Ten is",(44-4)/4)
print("Eleven is",(44/sqrt(4))/sqrt(4))
print("Twelve is",(math.factorial(4)-4-4-4))
print("Fourteen is",(4*4)-(4/sqrt(4)))
print("Fifteen is",(44/4)+4)
print("Sixteen is",((4**4)/4)/4)
print("Eigteen is",44-math.factorial(4)-sqrt(4))
print("Nineteen is",math.factorial(4)-(4/4)-4)
print("Twenty-one is",math.factorial(4)-(4/4)-sqrt(4))
print("Twenty-two is",44-math.factorial(4)+sqrt(4))
print("Twenty-four is",44-math.factorial(4)+4)




#
# examples of the most important four types of data
#   Numeric:     int, float
#   Sequential:  str, list
#

print("Our examples:")
print()

fav_num = 46      # an int
pi = 3.14         # a float

inator_creator = 'Dr. Heinz Doofenshmirtz'  # a string
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

print("Your examples:")

print()

fav_num = 3      # an int
rand_float = 3.33         # a float

my_name = 'IST341_Participant_7'  # a string
rand_list = ["food", 33, "august"]       # a list, of ints and strings

#
# now, let's print them -- and print their types:
#
print("fav_num is", fav_num, "\n  type(fav_num) is", type(fav_num))
print()  # prints a blank line, "\n" is another way to go to a newline

print("an example of a float is", rand_float, "\n  type(rand_float) is", type(rand_float))
print()

print("My name is", my_name, "\n  type(my_name) is", type(my_name))
print()

print("an example of a list is", rand_list, "\n  type(rand_list) is", type(rand_list))
print()




#
# example of clobbering fav_num
#

print("fav_num before is", fav_num)

fav_num = 1925            # Founded it!  (This was CGU's founding year.)

print("fav_num after is", fav_num)



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
# Your fair, complete RPS or RPS-variant  (this cell)
#
#
# Use the previous, unfair/incomplete examples as starting points and resources
# and keep adding/fixing/creating from there... .
#


import random       # imports a library (named random)
                    # for making random choices

user_name = input("Hello, you must be here to pass time or must be extremely bored! What is your name? ")
print("Welcome", user_name, ", whatever you're here for, please enjoy a classic game of rock-paper-scissors!")
print()

print("You will be playing against a computer who will choose rock, paper, or scissors at random.")
user_choice = input("Which do you choose? (rock/paper/scissors):")
comp_choice = random.choice(["rock","paper","scissors"])
print()

# We print both choices
print("You choose", user_choice)
print("I choose", comp_choice)
print()


# We can handle different possibilities with nested conditionals

if user_choice == "rock":

  if comp_choice == "rock":
    print("A tie?! This conversation is getting sedimental.")
  elif comp_choice == "paper":
    print("Yay! Paper will always beat rock!")
  else:
    print('Boo! Your rock completely demolished my scissors.')

elif user_choice == "paper":

  if comp_choice == "rock":
    print("Aww. You win...I'm sure I'll win next time.")
  elif comp_choice == "paper":
    print("Ugh. This is absolutely tear-able...we tied...")
  else:
    print('Ha! I win with my scissors. I guess I was feeling a bit snippy today.')

elif user_choice == "scissors":

  if comp_choice == "rock":
    print("I win! I'm CRUSHING this game.")
  elif comp_choice == "paper":
    print("I guess you win. I folded under pressure by choosing paper.")
  else:
    print('We tied!! That was not a cutting-edge move.')

else:

  print("You entered something I'm unfamiliar with...")
  print("... I'm sure it's something cool and super interesting...")



print()
print("Let's play again!")



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


# Title for this adventure: Music Adventure!
#
# Notes on how to "win" or "lose" this adventure:
#   To win, choose the green album or purple album.
#   To lose, choose the red album or yellow album.


def adventure():
    """This function runs one session of interactive fiction
       Well, it's "fiction," depending on the path chosen...
       Arguments: no arguments (prompted text doesn't count as an argument)
       Results: no results     (printing doesn't count as a result)
    """
    user_name = input("Hello, fellow music lover! Welcome to the Majestic Music Mountain! What should we call you?")

    if user_name:
      print()
      print("Welcome,", user_name, "!")
      print("Here, we listen to music 24/7. You can listen with your peers.")
      print("Or you can listen by yourself! Whatever experience you want!")
      print()

      print("However, in order to have access to all of the music in the world,")
      print("you must find and choose the correct music album at the top.")
      print("of the mountain.")
      print()

    music_player = input("Do you like to listen to music? [yes/no]")
    if music_player == "yes":
      print("Great! Music only makes any and every experience that much better!")
    else:
      print("Interesting choice...")
    print()

    print("On to the adventure!\n\n")
    print("As you start from the foot of the mountain,")
    print("you see groups of people listening to music on the stereo and")
    print("enjoying nature. You see people listening to music solo and")
    print("enjoying their own company.")
    print("Out of the corner of your eye, you see a stand")
    print("They are giving out headphones, speakers, and CD players.")
    print("You decide to head over and check one of the items out before you")
    print("make your way to the top.")

    choice1 = input("Do you choose the headphones, speakers, or CD players? [headphones/speakers/CD players]")
    print()

    if choice1 == "headphones":
      print("Awesome. A solo listener!")
    elif choice1 == "speakers":
      print("Whoever you choose to listen music with, I hope it's fun !")
    else:
      print("Good luck on your music quest!")
    print()

    print("Now, you are on your journey and you are listening to")
    print("all of the albums carefully.")
    print("As you're listening and listening, someone comes up to you.")
    print("They ask you if you want help choosing albums.")

    advice = input("Do you say yes or no to this person? [yes/no]")
    print()

    if advice == "yes":
      print("Trust them if you dare!")
    elif advice == "no":
      print("They could have been helpful...")

    print()

    print("After this encounter, you are super close to the top.")
    print("As you approach the stand where you need to make a choice,")
    print("the music master asks you what your choice will be.")

    choice2 = input("Do you choose the green, purple, red, or yellow album? [green/purple/red/yellow]")
    print()

    if choice2 == "green":
      print("You chose the green album!")
      print("What does this mean for your music-listening future?")
      print()
      print("...This means that...")
      print("you will forever have access to all of the music in the world!!!")
      print("Congratulations!!!")
    elif choice2 == "purple":
      print("You chose the purple album!")
      print("What does this mean for your music-listening future?")
      print()
      print("...This means that...")
      print("you will forever have access to all of the music in the world!!!")
      print("Congratulations!!!")
    elif choice2 == "red":
      print("You chose the red album!")
      print("What does this mean for your music-listening future?")
      print()
      print("...This means that...")
      print("you are unfortunately only limited to the music you know...")
      print("However, you can play again if you dare.")
    else:
      print("You chose the yellow album!")
      print("What does this mean for your music-listening future?")
      print()
      print("...This means that...")
      print("you are unfortunately only limited to the music you know...")
      print("However, you can play again if you dare.")

#
# The above adventure is defined in a _function_  (next week: functions!)
#
# This means it's defined, but it does not run until we call it.
# This line below calls it:
adventure()


