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


fav_num = input("What is your favorite number? ")

# Type Converting
next_num = int(fav_num) + 1

print("Your favorite number is", fav_num, "and one number bigger than your fav, is", next_num)



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

print("Zero is", 4+4-4-4)
print("One is", (4/4)*(4/4))  # yay!  exactly four fours - and no other digits


print("Two is", (4 / 4) + (4 / 4))
print("Three is", (4 + 4 + 4) / 4)
print("Four is", 4 + (4 - 4) * 4)
print("Five is", (4 * 4 + 4) / 4)
print("Six is", (4 + 4) / 4 + 4)
print("Seven is", (44 / 4) - 4)
print("Eight is", (4 + 4))
print("Nine is", (4/4 + 4 + 4))
print("Ten is", (44 / 4.4) )
print("Eleven is", (4/.4 + 4/4))
print("Twelve is", ((44 + 4) / 4))
print("Thirteen is", (44 / 4) + (4 / 4)) #############
print("Fourteen is", ((4 * (4 - .4)) - .4))
print("Fifteen is", (44 / 4 + 4))
print("Sixteen is", (4 + 4) + (4 + 4))
print("Seventeen is", (4 * 4) + (4 / 4))
print("Eighteen is", (44 * .4) + .4)
print("Nineteen is", (4 * 4) + 4 - 4 / 4) ############
print("Twenty is",  ((4 / 4 ) + 4) * 4)
print("Twenty-one is", (4.4 + 4) / .4)
print("Twenty-two is", (4 * 4) + 4 + (4 // 4))  ##########
print("Twenty-three is", (4 * 4) + 4 + (4 // 4) + 4 // 4)  ##########
print("Twenty-four is", (4 * 4) + 4 + 4)





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

Name = 'IST341_Participant_3'   #string
Age = 36   #int
Hight = 1.75   #float
Weight = 87   #float

fav_drinks = ["Coffee", "Tea", "Soda", "Water"]   #list

print("this is my Info:")
print("name:", Name, ", Age:", Age, ", Hight:", Hight, ", Weight:", Weight)
print("I like to drink: ", fav_drinks)
print()
print("OK, Now lets print Data Types:")
print(Name, "is a type of ", type(Name))
print(Age, "is a type of ", type(Age))
print(Hight, "is a type of ", type(Hight))
print(Weight, "is a type of ", type(Weight))
print(fav_drinks, "is a type of ", type(fav_drinks))






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

X = 6
Y = 4
Z = X + Y
print("Z = ", Z)
print()
X = Z + 1
print("Z = ", Z, ", X = ", X , ", Y = ", Y)



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

elif user_choice == "paper":
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
import random



user_name = input("What is your name? ")
print("Hello ,", user_name, " Welcome ")
print()

print("I challenge you to a fair game of RPS: rock-paper-scissors! Are you ready!")
print("You are SMART if you can beat ME, Can You ^^")
print("Get ready ")

user_choice = input("Which do you choose? (rock/paper/scissors)")
computer_choice = random.choice(["rock","paper","scissors"])

print("\n You chose: " ,user_choice, "and I chose: "  ,computer_choice)

if user_choice == computer_choice:
    print("Oh It's a tie!")
    print("You seem a worthy opponent")
    print("let us do it again")
elif user_choice == "rock":
    if computer_choice == "scissors":
        print("You win! Congratulations!")
    else:
        print("I win! Better luck next time!")
elif user_choice == "paper":
    if computer_choice == "rock":
        print("You win! Congratulations!")
    else:
        print("I win! Better luck next time!")
elif user_choice == "scissors":
    if computer_choice == "paper":
        print("You win! Congratulations!")
    else:
        print("I win! Better luck next time!")
else:
    print("Invalid choice. Please select rock, paper, or scissors. Check your spelling")

print()
print("Thanks for playing^^ I had good time playing with you")
print("If you are ready let's play again")


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


# Title for this adventure: "Conquering the Pyramids of Ancient Egypt"
#
##############################
# About this Game:
# You are an explorer venturing into the Pyramids of Giza in search of the fabled Jewel of Ra,
# said to grant eternal wisdom and power.
# To succeed, you must navigate traps, solve riddles, and choose wisely between ancient paths.
#############################
# Notes on how to "win" or "lose" this adventure:
#     To Win:
#     Choose "main" → "mural" → Answer "echo" → Final Answer "footsteps".
#  OR Choose "tunnel" → "staircase" → Duck to avoid arrows → Final Answer "footsteps".
#  OR Choose "tunnel" → "door" → Grab the Jewel → Final Answer "footsteps".
#############################

import random

def adventure():
    """An interactive fiction adventure: Conquering the Pyramids of Ancient Egypt."""

    print("Welcome to the Great Pyramid of Giza, brave explorer!")
    print("Your goal is to find the legendary Jewel of Ra and escape safely.")
    print()

    user_name = input("What is your name, adventurer? ")
    print("Welcome", user_name, "Prepare for a journey full of danger and mystery.")
    print()

    # First decision: How to enter the pyramid
    print("You stand before the Great Pyramid. You see two entrances:")
    print("1. The Main Entrance")
    print("2. A Hidden Tunnel")
    entrance = input("Do you choose the Main Entrance or the Hidden Tunnel? (main/tunnel): ")
    print()

    if entrance == "main":
        print("You enter through the Main Entrance. The air is heavy with ancient dust.")
        print("The passage is grand, but something feels ominous...")
        print()
    elif entrance == "tunnel":
        print("You crawl through the Hidden Tunnel. It's dark, but quiet.")
        print("You emerge into a dimly lit corridor.")
        print()
    else:
        print("Invalid choice! You hesitate too long, and a sandstorm forces you to retreat.")
        print("GAME OVER.")
        return

    # Second decision: Choose a path inside the pyramid
    print("Inside the pyramid, you see three paths:")
    print("1. A glowing mural with hieroglyphs.")
    print("2. A dark staircase descending into the abyss.")
    print("3. A golden door with the Eye of Horus.")
    path = input("Which path do you choose? (mural/staircase/door): ")
    print()

    if path == "mural":
        print("You approach the glowing mural. A booming voice asks you a riddle:")
        print("'I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I?'")
        print("Options: 1) Shadow  2) Echo  3) Whisper")
        answer = input("Choose your answer (1/2/3): ")
        if answer == "2":
            print("Correct! The mural reveals a secret passage.")
        else:
            print("Wrong! The mural traps you in its glow forever.")
            print("GAME OVER.")
            return
    elif path == "staircase":
        print("You descend the dark staircase. Suddenly, arrows shoot from the walls!")
        trap_action = input("Do you run forward or duck? (run/duck): ")
        if trap_action == "duck":
            print("You duck just in time and avoid the arrows. You proceed safely.")
        else:
            print("You are hit by the arrows and collapse.")
            print("GAME OVER.")
            return
    elif path == "door":
        print("You open the golden door and enter a room filled with treasure!")
        print("However, the floor begins to crumble beneath you.")
        treasure_action = input("Do you grab the Jewel of Ra and run, or retreat empty-handed? (grab/retreat): ")
        if treasure_action == "grab":
            print("You grab the Jewel of Ra and escape just in time. Victory!")
        else:
            print("You retreat safely, but the Jewel of Ra remains lost.")
            print("GAME OVER.")
            return
    else:
        print("Invalid choice! You wander aimlessly and fall into a hidden trap.")
        print("GAME OVER.")
        return

    # Final challenge: The Chamber of Ra
    print("You reach the final chamber, where the Path is guarded by a Sphinx.")
    print("The Sphinx challenges you with one last riddle:")
    print("'The more you take, the more you leave behind. What am I?'")
    print("Options: 1) Time  2) Footsteps  3) Sand")
    final_answer = input("Choose your answer (1/2/3): ")

    if final_answer == "2":
        print("Brilliant" , user_name, "You have solved the Sphinx's riddle.")
        print("The Jewel of Ra is yours, and you escape the pyramid!")
        print("CONGRATULATIONS! YOU WIN!")
    else:
        print("Wrong! The Sphinx curses you, and the pyramid seals you inside forever.")
        print("GAME OVER.")

# Start the adventure
adventure()




