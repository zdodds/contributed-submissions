# prompt: how to stop the code from runni


# prompt: how to stop the code from running in a code cell from a code cell

import os
import signal

# This function sends a SIGINT signal to the current process, effectively interrupting it.
def stop_current_process():
  os.kill(os.getpid(), signal.SIGINT)

# Example usage:  Uncomment to test
# stop_current_process()


2**3


2^3


2//3



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


fav_color = input("What is your favorite color?")


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
print("One is", 4/4*4/4)                 # Uh oh! No non-fours allowed!!
print("Two is", (4/4)+(4/4))   # Uh oh! This is incorrect... temporarily






-4 + sqrt(4) + factorial(4) - 4


#
#  Here is a space for your explorations and solutions
#       to the Four fours challenge.
#
# This will get you started.

import math

print("Zero is", int(4+4-4-4))
print("One is", int((4/4)*(4/4)))  # yay!  exactly four fours - and no other digits
print("Two is", int((4/4)+(4/4)))
print("Three is", int((4+4+4)/4))
print("Four is", int(math.sqrt(4)*math.sqrt(4)))
print("Five is", int(4+(math.sqrt(4)-(4/4))))
print("Six is", int(4+(math.sqrt(4))*(4/4)))
print("Seven is", int(4+(math.sqrt(4))+(4/4)))
print("Eight is", int(4+4+4-4))
print("Nine is", int(4+4+4/4))
print("Ten is", int(44/4-(4/4)))
print("Eleven is", int(44/4*4/4))
print("Twelve is", int(math.sqrt(4)*((4+4)-math.sqrt(4))))
print("Thirteen is", int(44/4+math.sqrt(4)))
print("Fourteen is", int(4*4-4/math.sqrt(4)))
print("Fifteen is", int(4*4-4/4))
print("Sixteen is", int(4*4*4/4))
print("Seventeen is", int(4*4+4/4))
print("Eighteen is", int(4*4+4/math.sqrt(4)))
print("Nineteen is", int(math.factorial(4)-(4+4/4)))
print("Twenty is", int(4*4+math.sqrt(4)+math.sqrt(4)))
print("Twenty-one is", int(math.factorial(4)-(4-4/4)))
#that weird floating decimal point makes me angry
#so I got rid of it
#yay ehehehehe :D


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

fav_food = "I have no idea"
print("fav_food is", fav_food, "\n  type(fav_food) is", type(fav_food))
print()

Sanrio_Characters = ['Cinnomoroll', 'My Melody', 'Kuromi', 'Pochacco', 'Pompompurin', 'Hello Kitty', 'and more!']
print("Sanrio_Characters are", Sanrio_Characters, "\n  type(Sanrio_Characters) is", type(Sanrio_Characters))
print()

Crewed_Moon_Landings = 6
print("Crewed_Moon_Landings is", Crewed_Moon_Landings, "\n  type(Crewed_Moon_Landings) is", type(Crewed_Moon_Landings))
print()

nice_number = 3.33
print("nice_number is", nice_number, "\n  type(nice_number) is", type(nice_number))
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
    print('Aargh! We tied but my rock was rockier!)')
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
#
# I made a rock paper scissors variant :D
# Yay! kskakgkvgxkvgkgksgv




import random

user_name = input("What is your name? ")
print("Hi, ", user_name, "! :D", sep='')
print()

def intro():
  """this is the intro/tutorial to the game"""
  print("the 'bird' eats the 'worm' and the 'fish',")
  print("the 'stone' impales the 'bird' and the 'fish' (it is a very sharp stone),")
  print("the 'worm' tunnels through the 'stone' and the eats the 'plant',")
  print("the 'fish' eats the 'plant' and the 'worm',")
  print("and the 'plant' traps the 'bird' and cracks the 'rock'")
  print()

print("Let's play bird-stone-worm-fish-plant!")
understand = input ("Do you know how to play? (yes, no) ")
print()
if understand == "no":
  print("Here,")
  intro()
  understand = input ("Do you understand now? (yes, no) ")

  if understand == "no":
    print("*sigh*")
    print("Fine, let me explain it one more time")
    print()
    intro()
    print("I will assume you understand now")
  else:
    print("Great!")

else:
  print("Great!")

def le_game():
  """le game"""
  user_choice = input("Which do you choose? (bird/stone/worm/fish/plant):").lower()
  comp_choice = random.choice(["bird","stone","worm", "fish", "plant"])
  print("I see, you chose", user_choice)
  print("and I chose", comp_choice)
  print()

  if user_choice == "bird":
    if comp_choice == "bird":
      print("My bird and your bird say hi to each other.")
      print("It's a tie!")
    elif comp_choice == "stone":
      print("My stone flies into your bird, knocking it down.")
      print("Yes! I win! Better luck next time, ehehe.")
    elif comp_choice == "worm":
      print("My worm was enjoying its little wormy day until your bird swooped down.")
      print("No! I lost! :( Good job!")
    elif comp_choice == "fish":
      print("My fish was swimming down the river when your bird decided it wanted a snack!")
      print("No! I lost! :( Good job!")
    else:
      print("My big planty vines snuck up behind your bird and pinned down its wings!")
      print("Yes! I win! Better luck next time, ehehe.")

  elif user_choice == "stone":
    if comp_choice == "bird":
      print("My bird was flying until your stone smacked it in the face!")
      print("No! I lost! :( Good job!")
    elif comp_choice == "stone":
      print("Both of our stones sit there. The end.")
      print("It's a tie!")
    elif comp_choice == "worm":
      print("My worm encountered your stone and tunneled right through it!")
      print("Yes! I win! Better luck next time, ehehe.")
    elif comp_choice == "fish":
      print("My fish swam right into your stone!")
      print("No! I lost! :( Good job!")
    else:
      print("My plant sprouted up right in the middle of your stone!")
      print("Yes! I win! Better luck next time, ehehe.")

  elif user_choice == "worm":
    if comp_choice == "bird":
      print("My bird decided it was hungry and picked your worm up for a snak!")
      print("Yes! I win! Better luck next time, ehehe.")
    elif comp_choice == "stone":
      print("My stone was minding its own business when your worm came and tunneled through it...")
      print("No! I lost! :( Good job!")
    elif comp_choice == "worm":
      print("Both of our worms decide to chomp some dirt together.")
      print("It's a tie!")
    elif comp_choice == "fish":
      print("My fish saw your worm and gobbled it up!")
      print("Yes! I win! Better luck next time, ehehe.")
    else:
      print("My planty weed became your worm's dinner.")
      print("No! I lost! :( Good job!")

  elif user_choice == "fish":
    if comp_choice == "bird":
      print("My bird noticed your fish and decided it was hungry.")
      print("Yes! I win! Better luck next time, ehehe.")
    elif comp_choice == "stone":
      print("My stone blocked your fish's path, and wasn't noticed until it was too late...")
      print("Yes! I win! Better luck next time, ehehe.")
    elif comp_choice == "worm":
      print("My worm wriggled straight into your fish's mouth!")
      print("No! I lost! :( Good job!")
    elif comp_choice == "fish":
      print("Our fish see each other and become friends.")
      print("It's a tie!")
    else:
      print("My very planty seaweed's very planty existence was cut short by your fish.")
      print("No! I lost! :( Good job!")

  elif user_choice == "plant":
    if comp_choice == "bird":
      print("My bird flew right into your planty tree's branches!")
      print("No! I lost! :( Good job!")
    elif comp_choice == "stone":
      print("My stone was very rudely cracked in half by your planty weed.")
      print("No! I lost! :( Good job!")
    elif comp_choice == "worm":
      print("My worm chomped a whole hole through your planty leaf.")
      print("Yes! I win! Better luck next time, ehehe.")
    elif comp_choice == "fish":
      print("My fish took a bite out of your planty plant!")
      print("Yes! I win! Better luck next time, ehehe.")
    else:
      print("Both of our plants look for the sunlight.")
      print("It's a tie!")

  else:
    print("It looks like what you chose doesn't make sense...")
    print("Please choose again.")
    print("AND MAKE SURE YOU CHOOSE RIGHT THIS TIME!!! >:(")
    print()
    le_game()

  print()
  play_again = input("Play again? (yes/no)")
  if play_again == "yes":
    play_again = "h"
    le_game()

  else:
    print ("Ok! Bye!")

le_game()


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


# Title for this adventure: The Flower Garden
#
# To win,
# Don't input anything random more than 3 times
# And choose clematis and asphodel
# (After the intro input 1 then 2)
# To lose,
# Choose lilac or applethorn or input random things 4 or more times




def my_adventure():
    """my adventure
    (obviously)"""
    global strikes
    strikes = 0
    clematis = 0
    asphodel = 0
    last_name = 'nothing! :D'

    def lose():
      print()
      print('You lose, sorry :(')
      playagain = input("Play again? (yes/no) ")
      if playagain == "yes":
        from IPython.display import clear_output
        clear_output()
        print('restarting...')
        my_adventure()
      else:
        print("Bye!")
        def stop_current_process():
          import os
          import signal
          os.kill(os.getpid(), signal.SIGINT)
        stop_current_process()

    def strike():
      """don't do this or you lose !"""
      if strikes == 1:
        print("No, no, no! That's not an option!")
        print("Make sure you only type the number.")
        print("No dot or space or anything else.")
        print("This is your first strike, so I'll let you off this time.")
        print("Be careful, though. 3 strikes and you lose!")
        print()
      elif strikes == 2:
        print("Be careful!")
        print("3 strikes and you lose!")
        print("This is strike 2.")
        print()
      elif strikes == 3:
        print("Be careful!")
        print("3 strikes and you lose!")
        print("This is the last strike.")
        print()
      else:
        print("You've done this too many times! >:(")
        lose()




    user_name = input("What is the moniker that you go by? (First name please) ")
    if user_name.lower() == "zach" or user_name.lower() == "zachary":
      print("Oh!")
      last_name = input("What be your surname? ")
      if last_name.lower() == "dodds":
        print("Aha!")
        print()
        print("...You see a shadowy figure walk towards you.")
        print("It says 'You like poptarts, right?")
        print("You nod.")
        print("'Then join the Cult of Flowers and Poptarts! ehehehehe'")
        print("It slinks away.")
        print()
        print("Odd.")
        print()
        print("You feel a sudden urge to eat a poptart.")
      else:
        print("Oh.")
        print("Sorry, I thought you were someone else.")
    else:
      print("Hmmm... Where have I heard that name before? \n...")
    print()
    print("Anyways")
    print("Hello, ", user_name, "!", sep='')
    print("I am the narrator.")
    print('I suspect you will be hearing me a lot.')
    print()
    def choice1():
      choices = input("1. So you're not just a voice in my head? \n2. Hello there, narrator \n(input 1 or 2!) \n ")
      if choices == "1":
        print("*gasp* Of course I'm not!")
      elif choices == "2":
        print("Hello to you too! (I like you already! :D)")
      else:
        global strikes
        strikes = int(strikes) + 1
        strike()
        choice1()
    choice1()

    print("Now that I know you, let's begin!")
    input("Input anything to begin!")

    from IPython.display import clear_output
    clear_output()

    print("Welcome to the Flower Garden.")
    print("There are many flowers in this garden.")
    print("Your goal is to find the Red Camellia.")
    print("Use your knowledge of Victorian Era flower meanings to find it!")
    print("Good luck!")
    print("~~~~~~~~~~~~~~~~~~")
    print("You find yourself in a clearing surrounded by small stone walls.")
    print("There is a fence in one of the four walls.")
    print("That must be where you are supposed to go.")
    print("You see two large garden beds, both filled with purple flowers.")
    print("On the left are clematises, the right, lilacs.")
    print("Between them, there is a table. Upon the table, there is a letter and a dictionary of flowers.")
    print("You pocket the dictionary before reading the letter.")
    print("It says:")
    print("       The key lies in the mind, but remembering times past will only trap you.")
    print()

    def choice2():
      print("BE CAREFUL WITH YOUR CHOICES!!!")
      print("INVESTIGATING THE WRONG THING HAS CONSEQUENCES.")
      clematis = input("What do you do now? (1/2/3/4) \n 1. Investigate the clematises \n 2. Investigate the lilacs \n 3. Look up clematis in the flower dictionary \n 4. Look up lilac in the flower dictionary \n ")
      print()
      if clematis == '1':
        print("You investigate the clematises.")
        print("Inside the largest flower, there is a shiny purple key.")
        print("You take the key and walk over to the fence.")
        print("With bated breath, you insert and turn the key.")
        print("The fence swings open.")
        print()
        print()
      elif clematis == '2':
        print("You investigate the lilacs.")
        print("When you touch the petals if one of the lilacs, you begin to feel dizzy.")
        print("Flashes of color appear in your vision, eventually becoming what seems to be...")
        print("memories.")
        print("Memories from when you didn't even know you could remember.")
        print("Both good and bad.")
        print("You don't even realize when you pass out until you wake up in your bed.")
        print("Had it been a dream all along?")
        lose()
      elif clematis == '3':
        print("You look up clematis in the dictionary.")
        print("The dictionary says:")
        print("       The clematis is a flowering vine that represents \n mental acuity, cleverness, and ingenuity.")
        print()
        choice2()
      elif clematis == '4':
        print("You look up lilac in the dictionary.")
        print("The dictionary says:")
        print("       The lilac is a flowering bush that represents old love, \n renewal, and rememberance.")
        print()
        choice2()
      else:
        global strikes
        strikes = int(strikes) + 1
        strike()
        choice2()

    choice2()
    print("You step through the gates and find yourself in another area identical to the last.")
    print("The only difference is that the flowers are pure white.")
    print("The applethorn and the asphodel.")
    print("The fence gates swing shut behind you with a loud boom.")
    print("Turning around, you see that it dissolved into the wall.")
    print("Oh well. You aren't going to go back there again.")
    print("Turning back around, you see a beautiful arched doorway at the far end.")
    print("That must be the way out.")
    print("Once again, there is a letter for you to read.")
    print("It says:")
    print("       Fear not death, but the untrustworthy. Fear not regrets, but \n the outwardly charming.")
    print()

    def choice3():
      print("BE CAREFUL WITH YOUR CHOICES.")
      print("INVESTIGATING THE WRONG THING HAS C O N S E Q U E N C E S!!!")
      asphodel = input("What do you do now? \n 1. Investigate the applethorns \n 2. Investigate the asphodels \n 3. Look up applethorn in the flower dictionary \n 4. Look up asphodel in the flower dictionary \n ")
      print()
      if asphodel == '1':
        print("You investigate the applethorns.")
        print("As you get closer to the applethorns, it's sweet scent allures you.")
        print("A tingling feeling tells you this is dangerous, but you can't seem to move away.")
        print("You touch a flower.")
        print("The scent of the applethorns engulf you, and you faint.")
        print("You wake up in your bed.")
        print("Had it been a dream all along?")
        lose()
      elif asphodel == '2':
        print("You investigate the asphodels.")
        print("The first asphodel you touch wilts beneath your fingers.")
        print("You behin to feel apprehensive.")
        print("Then, the flower snaps off, and you can see a shiny thing in the hollow stem.")
        print("It turns out to be a gleaming silver key, delicate and small.")
        print("You pick it up and take it over to the door, careful not to snap it.")
        print("The key slides in smoothly, turning with a satisfying click.")
        print("You turn the doorknob, and the doors open.")
        print()
        print()
      elif asphodel == '3':
        print("You look up applethorn in the dictionary.")
        print("The dictionary says:")
        print("       The applethorn is a poisonous flowering plant that symbolizes \n deceitful charm.")
        choice3()
      elif asphodel == '4':
        print("You look up asphodel in the dictionary.")
        print("The dictionary says:")
        print("       The asphodel is a lily that symbolizes death, literally \n meaning 'My regrets follow you to the grave.'")
        choice3()
      else:
        global strikes
        strikes = int(strikes) + 1
        strike()
        choice3()
    choice3()

    print("The garden layout is about the same as before, except for one thing.")
    print("Well, a few things.")
    print("It is filled with brilliant red flowers, and in the center, a deep crimson flowerbush grows.")
    print("It is the Red Camellia.")
    print("You hesitantly step forward and touch it, and a warm feeling sprouts in your chest.")
    print("You feel like you can perservere through the darkest of times.")
    print("That what you hold dear will never be separated from you.")
    print("The world fades to black.")
    print()
    print("When you wake up, you realize it had been a dream.")
    print("But it had been a good one.")
    print()
    print("~~~~~~~~~~~~~~~~~~")
    print("You win!")
    print("Good job :D")
    if last_name.lower() != "dodds":
      print("(P.S., the advertiser didn't stop by, but join the Cult of Flowers and Poptarts!! ^u^)")






#
# The above adventure is defined in a _function_  (next week: functions!)
#
# This means it's defined, but it does not run until we call it.
# This line below calls it:
my_adventure()


