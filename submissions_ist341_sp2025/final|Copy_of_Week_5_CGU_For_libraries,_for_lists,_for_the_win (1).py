L = [ 'CGU', 'CMC', 'PIT', 'SCR', 'POM', 'HMC' ]
print("len(L) is", len(L))
print(max(L), min(L))     # just for fun, try max and min of this L:  We win! (Why?!)


L = [1, 2, 40, 3 ]
print("max(L) is", max(L))
print("min(L) is", min(L))
print("sum(L) is", sum(L))


L = range(1,43)
print("L is", L)   # Uh oh... it won't create the list unless we tell it to...


L = list(range(1,367))  # ask it to create the list values...
print("L is", L)  # Aha!


print("max(L) is", max(L))    # ... up to but _not_ including the endpoint!


#
# Gauss's number: adding from 1 to 100
#
print(sum(list(range(101))))


# single-character substitution:

def vwl_once(c):
  """ vwl_once returns 1 for a single vowel, 0 otherwise
  """
  if c in 'aeiou': return 1
  else: return 0

# two tests:
print("vwl_once('a') should be 1 <->", vwl_once('a'))
print("vwl_once('b') should be 0 <->", vwl_once('b'))


s = "claremont"
print("s is", s)
print()

LC = [ vwl_once(c) for c in s ]
print("LC is", LC)


s = "audio"
print("s is", s)
print()

LC = [ vwl_once(c) for c in s ]
print("LC is", LC)


#
# feel free to use this cell to check your vowel-patterns.
#      This example is a starting point.
#      You'll want to copy-paste-and-change it:
s = "state"
LC = [ vwl_once(c) for c in s ]
print("LC is", LC)

s = "shrink"
LC = [ vwl_once(c) for c in s ]
print("LC is", LC)

s = "amounted"
LC = [ vwl_once(c) for c in s ]
print("LC is", LC)

s = "manipulative"
LC = [ vwl_once(c) for c in s ]
print("LC is", LC)

s = "strengths"
LC = [ vwl_once(c) for c in s ]
print("LC is", LC)



#
# vwl_all using this technique
#

def vwl_all(s):
  """ returns the total # of vowels in s, a string
  """
  LC = [ vwl_once(c) for c in s ]
  total = sum(LC)  # add them all up!
  return total

# two tests:
print("vwl_all('claremont') should be 3 <->", vwl_all('claremont'))
print("vwl_all('caffeine') should be 4 <->", vwl_all('caffeine'))



# scrabble-scoring

def scrabble_one(c):
  """ returns the scrabble score of one character, c
  """
  c = c.lower()
  if c in 'alienorstu':   return 1
  elif c in 'dg':         return 2
  elif c in 'bcmp':       return 3
  elif c in 'fhvwy':      return 4
  elif c in 'k':          return 5
  elif c in 'jx':         return 8
  elif c in 'qz':         return 10
  else:                   return 0

# tests:
print("scrabble_one('q') should be 10 <->", scrabble_one('q'))
print("scrabble_one('!') should be 0 <->", scrabble_one('!'))
print("scrabble_one('u') should be 1 <->", scrabble_one('u'))


#
# scrabble_all using this technique
#

def scrabble_all(s):
  """ returns the total scrabble score of s
  """
  LC = [ scrabble_one(c) for c in s ]
  total = sum(LC)  # add them all up!
  return total


# two tests:
print("scrabble_all('Zany Sci Ten Quiz') should be 46 <->", scrabble_all('Zany Sci Ten Quiz'))
print("scrabble_all('Claremont') should be 13 <->", scrabble_all('Claremont'))
print("scrabble_all('abcdefghijklmnopqrstuvwxyz!') should be 87 <->", scrabble_all('abcdefghijklmnopqrstuvwxyz!'))


# Here are the two texts:

PLANKTON = """I'm As Evil As Ever. I'll Prove It
Right Now By Stealing The Krabby Patty Secret Formula."""

PATRICK = """I can't hear you.
It's too dark in here."""


#
# raw scrabble comparison
#

print("PLANKTON, total score:", scrabble_all(PLANKTON))
print("PATRICK, total score:", scrabble_all(PATRICK))


#
# per-character ("average/expected") scrabble comparison
#

print("PLANKTON, per-char score:", scrabble_all(PLANKTON)/len(PLANKTON))
print("PATRICK, per-char score:", scrabble_all(PATRICK)/len(PATRICK))



# let's see a "plain"  LC   (list comprehension)

[ 2*x for x in [0,1,2,3,4,5] ]

# it _should_ result in     [0, 2, 4, 6, 8, 10]



# let's see a few more  list comprehensions.  For sure, we can name them:

A  =  [ 10*x for x in [0,1,2,3,4,5] if x%2==0]   # notice the "if"! (there's no else...)
print(A)



B = [ y*21 for y in list(range(0,3)) ]    # remember what range does?
print(B)





C = [ s[1] for s in ["hi", "7Cs!"] ]      # doesn't have to be numbers...
print(C)



# Let's try thinking about these...

"""
A = [ n+2 for n in   range(40,42) ]
B = [ 42 for z in [0,1,2] ]
C = [ z for z in [42,42] ]
D = [ s[::-2] for s in ['fungic','y!a!y!'] ]
L = [ [len(w),w] for w in  ['Hi','IST'] ]

A: [42, 43]
B; [43, 43, 43]
C: [42, 42]
D: ['cgu', '!!!']
L: [['2', 'hi'], ['3', 'IST']]
"""

# then, see if they work the way you predict...
A = [ n+2 for n in   range(40,42) ]
B = [ 42 for z in [0,1,2] ]
C = [ z for z in [42,42] ]
D = [ s[::-2] for s in ['fungic','y!a!y!'] ]
L = [ [len(w),w] for w in  ['Hi','IST'] ]

print(A, '\n' , B, '\n', C, '\n', D, '\n', L, sep='')


#
# here, [1] create a function that scores punctuation as 1 and non-punctuation as 0, pun_one
#       [2] use the list-comprehension technique to create a punctuation-scoring function, pun_all

# use the above examples as a starting point!
def pun_one(c):
  if c in '.,(){}[]?!-–—\'":;…':
    return 1
  else:
    return 0


def pun_all(s):
  LC = [pun_one(c) for c in s]
  return sum(LC)





# tests
print("pun_all(PLANKTON) should be 4 <->", pun_all(PLANKTON))
print("pun_all(PATRICK) should be 4 <->", pun_all(PATRICK))
# They're more similar, punctuation-wise than scrabble-wise


#
# The two works of yours and two of another author's ("theirs")
#     These can - and should - be large
#

YOURS1 = """ I would have pulled away then, if I could, but small, firm, fingers pulled me forward unrelentingly into the dark.
     “You should always be prepared for the unexpected.” He said. “Otherwise the Click Clacks will come for you.”
     “Well, how am I supposed to prepare for the unexpected if it’s, well,  unexpected?”
     I could feel, but not see, the boy’s hand move as he shrugged. “I don’t know. The unexpected should be expected, no?”
     Click. Clack.
     I wondered how children even came up with these things. I certainly wouldn’t have been able to come up with something so illogical and far-fetched. At least they definitely aren’t lacking in creativity.
     The boy had sat down on the bed now, pulling me down with him. I asked, “Then, how come everyone hasn’t been “drank” by the Click Clacks? I’m sure not everyone always expects the unexpected.”
     “They just got lucky.” He said ominously.
     I couldn’t think of anything else to say, so I let the silence drape over us like a curtain of darkness. It was quiet. Too quiet.
     Click. Clack.
     I blinked and tried to will away the thought of the Click Clacks. The house was old, and the click clacks were probably just the creaks of the old house in the wind. The story had made me more apprehensive than I would admit. It was even making me hear things.
     I felt a prickle along my spine, like spiders crawling. I had a feeling the boy was staring at me, even in the pitch blackness of the attic. There was a rustle as he moved, then I could feel his breath against my ear.
     “But you didn’t.”
     It was but a whisper, but it chilled me to my core. My heart pounded, and I couldn’t say a word. Reaching out towards the boy, I tried to grab him, but my hand just passed through air. The darkness seemed even more suffocating now.
     Click. Clack.
     I heard the tip-tap of footsteps, deafening in the silence. They came from all around me, cocooning me with the roar. It almost sounded like millions of spiders crawling towards me. I felt a huge presence emanating from behind me, but I couldn’t turn around. Even if I did, I wouldn’t have been able to see what it was. My instincts screamed at me to run, to hide, to do something to escape. But I knew I wouldn’t be able to make it through the door. I knew I had to stay still. I knew I wouldn’t be able to escape anyways.
     Click. Clack.
     Thousands of tiny eyes surrounded me, barely noticeable. The moment seemed to drag out for an eternity. Anticipation flooded my mind with horrifying images. My heart pounded. My eyes strained. My stomach roiled and I was hyper aware of every sound, every touch, every glint of light reflecting off the eyes. Then, I heard a whoosh. And the moment shattered.
     The thing grabbed my arm just as I tried to get up and flee. It pulled me towards it and the spiders, yes, the eyes were spiders, closing in on me. I struggled. I twisted. I tried to bite. But nothing could get me out of its cold embrace.
     Click. Clack.
     The lights flickered on for a moment, and I saw a dark shadow, vaguely humanoid. Shadowy tendrils that were the spiders radiated off of it. That shadow was the thing. That, I now knew for certain, was a Click Clack. The lights flickered off again and I felt the Click Clack reach down its ‘head’. A bite. A scream. It hurt. I was scared. I was terrified. Light flooded the room. Then, all I could see was darkness.

"""
# click clack the rattlebag my ending
YOURS2 = """ Mahmoud opened his eyes, still burning from the tear gas, and looked around. The Hungarian soldiers were still there, but the crowd had dispersed. He spotted many people heading away from the border, and those that didn’t, were taken by the soldiers to who-knows-where. Scanning his surroundings, he looked for Waleed and his parents, but they were nowhere to be seen.
     There! He saw them, unconscious, being taken into Hungary by the soldiers. Mahmoud stood up with shaky knees and ran over, shouting “Wait! Please! Where are you taking them?”
     As he approached, a soldier pushed him away, making Mahmoud stumble. “Why don’t you just let us go through Hungary? We don’t even want to stay! Please!” He shouted.
     The soldier sighed, as if exasperated, and grabbed Mahmoud, pulling him to where they were taking his family. He noticed Waleed twitch, signaling that he was probably waking up.
     “Waleed! Can you try to wake Mom and Dad up? I don’t know where they’re taking us!”
     Waleed just shrugged, nodding towards his bound arms. “Who knows? Maybe they’re taking us to Austria.”
     Mahmoud glared at his brother. “Do you think they would willingly help us?” He only received a halfhearted grunt in response. Defeated, Mahmoud struggled against the soldier who was half dragging, half leading him, but the soldier was far stronger and healthier than Mahmoud.
     He ignored Mahmoud and led him towards a group of people who were also being detained. They were tied up and thrown into a van, and so was Mahmoud’s family. Struggling again, Mahmoud tried to free his hands unsuccessfully. He was manhandled into the van, and behind him, the door closed. Through the small window, he could see his surroundings move. They were headed back over the border.
     Over the border, out of Hungary, and away from Germany.

"""
# when we were reading refugee

THEIRS1 = """ AFTER THE MOVIE was over it suddenly came to us that Cherry and Marcia
didn't have a way to get home. Two-Bit gallantly offered to walk them home--- the west
side of town was only about twenty miles away--- but they wanted to call their parents
and have them come and get them. Two-Bit finally talked them into letting us drive them
home in his car. I think they were still half-scared of us. They were getting over it,
though, as we walked to Two-Bit's house to pick up the car. It seemed funny to me that
Socs--- if these girls were any example--- were just like us. They liked the Beatles and
thought Elvis Presley was out, and we thought the Beatles were rank and that Elvis was
tuff, but that seemed the only difference to me. Of course greasy girls would have acted a
lot tougher, but there was a basic sameness. I thought maybe it was money that separated
us.
"No," Cherry said slowly when I said this. "It's not just money. Part of it is, but
not all. You greasers have a different set of values. You're more emotional. We're
sophisticated--- cool to the point of not feeling anything. Nothing is real with us. You
know, sometimes I'll catch myself talking to a girl-friend, and realize I don't mean half of
what I'm saying. I don't really think a beer blast on the river bottom is super-cool, but I'll
rave about one to a girl-friend just to be saying something." She smiled at me. "I never
told anyone that. I think you're the first person I've ever really gotten through to."
She was coming through to me all right, probably because I was a greaser, and
younger; she didn't have to keep her guard up with me.
"Rat race is a perfect name for it," she said. "We're always going and going and
going, and never asking where. Did you ever hear of having more than you wanted? So
that you couldn't want anything else and then started looking for something else to want?
It seems like we're always searching for something to satisfy us, and never finding it.
Maybe if we could lose our cool. we could."
"""

THEIRS2 = """ I WOKE UP LATE IN the afternoon. For a second I didn't know where I was.
You know how it is, when you wake up in a strange place and wonder where in the world
you are, until memory comes rushing over you like a wave. I half convinced myself that I
had dreamed everything that had happened the night before. I'm really home in bed, I
thought. It's late and both Darry and Sodapop are up. Darry's cooking breakfast, and in a
minute he and Soda will come in and drag me out of bed and wrestle me down and tickle
me until I think I'll die if they don't stop. It's me and Soda's turn to do the dishes after we
eat, and then we'll all go outside and play football. Johnny and Two-Bit and I will get
Darry on our side, since Johnny and I are so small and Darry's the best player. It'll go like
the usual weekend morning. I tried telling myself that while I lay on the cold rock floor,
wrapped up in Dally's jacket and listening to the wind rushing through the trees' dry
leaves outside.
Finally I quit pretending and pushed myself up. I was stiff and sore from sleeping
on that hard floor, but I had never slept so soundly. I was still groggy. I pushed off
Johnny's jeans jacket, which had somehow got thrown across me, and blinked, scratching
my head. It was awful quiet, with just the sound of rushing wind in the trees. Suddenly I
realized that Johnny wasn't there.
"Johnny?" I called loudly, and that old wooden church echoed me, onny onny... I
looked around wildly, almost panic-stricken, but then caught sight of some crooked
lettering written in the dust of the floor. Went to get supplies. Be back soon. J.C.
I sighed, and went to the pump to get a drink. The water from it was like liquid ice
and it tasted funny, but it was water. I splashed some on my face and that woke me up
pretty quick. I wiped my face off on Johnny's jacket and sat down on the back steps. The
hill the church was on dropped off suddenly about twenty feet from the back door, and
you could see for miles and miles. It was like sitting on the top of the world.
"""
# reading the outsiders for english right now...
# WHY IS EVERYTHING HERE FROM E N G L I S H


len(THEIRS2)


#
# Here, run your punctuation-comparisons (absolute counts)
#

print("pun_all(YOURS1) is", pun_all(YOURS1))
print("pun_all(YOURS2) is", pun_all(YOURS2))
print("pun_all(THEIRS1) is", pun_all(THEIRS1))
print("pun_all(THEIRS2) is", pun_all(THEIRS2))


#
# Here, run your punctuation-comparisons (relative, per-character counts)
#
print("punctuation per character in YOURS1 is", pun_all(YOURS1)/len(YOURS1))
print("punctuation per character in YOURS2 is", pun_all(YOURS2)/len(YOURS2))
print("punctuation per character in THEIRS1 is", pun_all(THEIRS1)/len(THEIRS1))
print("punctuation per character in THEIRS2 is", pun_all(THEIRS2)/len(THEIRS2))


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
        #print("I guess", guess)  # comment this out - avoid printing when analyzing!
        guess = choice( range(0,100) )  # 0 to 99, inclusive
        number_of_guesses += 1

    return number_of_guesses

# test our function!
guess(42)


# Let's run 10 number-guessing experiments!

L = [ guess(42) for i in range(10) ]
print(L)

# 10 experiments: let's see them!!


# Let's see what the average results were
print("len(L) is", len(L))
ave = sum(L)/len(L)
print("average is", ave )

# try min and max, count of 42's, count of 92's, etc.
print("min(L) is", min(L), "and max(L) is", max(L))
has3 = ['3' for i in L if '3' in str(i)]
print("number of numbers with 3 is", len(has3))


#
# Let's try again... with the dice-rolling experiment
#

from random import choice

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

        #print("run", i, "roll:", d1, d2, you, flush=True)
        #time.sleep(.01)

    return numdoubles

# test our function!
count_doubles(300)


L = [ count_doubles(300) for i in range(1000) ]
print("doubles-counting: L[0:5] are", L[0:5])
print("doubles-counting: L[-5:] are", L[-5:])
#
# Let's see what the average results were
print("len(L) is", len(L))
ave = sum(L)/len(L)
print("average is", ave )

# try min and max, count of 42's, count of 92's, etc.
print("min(L) is", min(L), "and max(L) is", max(L))
more = ['m' for i in L if i > 60]
print('# of #\'s more than 60 is', len(more))



# let's try our birthday-room experiment:

from random import choice

def birthday_room( days_in_year = 365 ):    # note: default input!
    """
        run the birthday room experiment once!
    """
    B = []
    next_bday = choice( range(0,days_in_year) )

    while next_bday not in B:
        B += [ next_bday ]
        next_bday = choice( range(0,days_in_year) )

    B += [ next_bday ]
    return B



# test our three-curtain-game, many times:
result = birthday_room()   # use the default value
print(result)



L = [ len(birthday_room()) for i in range(1000) ]
print("birthday room: L[0:5] are", L[0:5])
print("birthday room: L[-5:] are", L[-5:])
#
# Let's see what the average results were
print("len(L) is", len(L))
ave = sum(L)/len(L)
print("average is", ave )

# try min and max, count of 42's, count of 92's, etc.
print("min(L) is", min(L), "and max(L) is", max(L))
ten = ['t' for i in L if i < 10]
print('less than 10:', len(ten))
print('percentage less than 10:', (len(ten)/len(L))*100, '%')
fifty = ['f' for i in L if i > 50]
print('more than 50:', len(fifty))
print('percentage more than 50:', (len(fifty)/len(L))*100, '%')


#
# Example Monte Carlo simulation: the Monte-Carlo Monte Hall paradox
#

from random import choice

def count_wins( N, original_choice, stay_or_switch ):
    """
        run the Monte Hall paradox N times, with
        original_choice, which can be 1, 2, or 3 and
        stay_or_switch, which can be "stay" or "switch"
        Count the number of wins and return that number.
    """
    numwins = 0       # start with no wins so far...

    for i in range(1,N+1):      # run repeatedly: i keeps track
        win_curtain = choice([1,2,3])   # the curtain with the grand prize
        original_choice = original_choice      # just a reminder that we have this variable
        stay_or_switch = stay_or_switch        # a reminder that we have this, too

        result = ""
        if original_choice == win_curtain and stay_or_switch == "stay": result = " Win!!!"
        elif original_choice == win_curtain and stay_or_switch == "switch": result = "lose..."
        elif original_choice != win_curtain and stay_or_switch == "stay": result = "lose..."
        elif original_choice != win_curtain and stay_or_switch == "switch": result = " Win!!!"

        #print("run", i, "you", result, flush=True)
        #time.sleep(.025)

        if result == " Win!!!":
            numwins += 1


    return numwins

# test our three-curtain-game, many times:
count_wins(300, 1, "stay")



L = [ count_wins(300,1,"stay") for i in range(1000) ]
print("curtain game: L[0:5] are", L[0:5])
print("curtain game: L[-5:] are", L[-5:])
#
# Let's see what the average results were
print("len(L) is", len(L))
ave = sum(L)/len(L)
print("average is", ave )

# try min and max, count of 42's, count of 92's, etc.
print("min(L) is", min(L), "and max(L) is", max(L))
third = ['h' for i in L if i < 100]
print('less than a third won:', len(third))
print('percentage less than a third won:', (len(third)/len(L))*100, '%')
half = ['f' for i in L if i > 150]
print('more than half won', len(half))
print('percentage more than half won:', (len(half)/len(L))*100, '%')


#
# First, the random-walking code:
#

import random

def rs():
    """One random step"""
    return random.choice([-1, 1])

def rwalk(radius):
    """Random walk between -radius and +radius  (starting at 0 by default)"""
    totalsteps = 0          # Starting value of totalsteps (_not_ final value!)
    start = 0               # Start location (_not_ the total # of steps)

    while True:             # Run "forever" (really, until a return or break)
        if start == -radius or start == radius:
            return totalsteps # Phew!  Return totalsteps (stops the while loop)

        start = start + rs()
        totalsteps += 1     # addn totalsteps 1 (for all who miss Hmmm :-)

        # print("at:", start, flush=True) # To see what's happening / debugging
        # ASCII = "|" + "_"*(start- -radius) + "S" + "_"*(radius-start) + "|"
        # print(ASCII, flush=True) # To see what's happening / debugging

    # it can never get here!

# Let's test it:
rwalk(5)   # walk randomly within our radius... until we hit a wall!



# Analyze!
# create List Comprehensions that run rwalk(5) for 1000 times

# Here is a starting example:
L = [ rwalk(5) for i in range(1000) ]

# then, find the average of those 1000 experiments for rwalk(5)
average = sum(L)/len(L)
print("The average for radius==5 (for 1000 trials) was", average)


# Next, try it for more values...
# Then, you'll create a hypothesis about what's happening!




# Repeat the above for a radius of 6, 7, 8, 9, and 10
# It's fast to do:  Copy and paste and edit!!

L = [ rwalk(5) for i in range(1000) ]
average = sum(L)/len(L)
print("The average for radius==5 (for 1000 trials) was", average)


L = [ rwalk(6) for i in range(1000) ]
average = sum(L)/len(L)
print("The average for radius==6 (for 1000 trials) was", average)


#
# see if we have the requests library...
#

import requests


#
# let's try it on a simple webpage
#

#
# we assign the url and obtain the api-call result into result
#    Note that result will be an object that contains many fields (not a simple string)
#

url = "https://www.cs.hmc.edu/~dodds/demo.html"
result = requests.get(url)

# if it succeeded, you should see <Response [200]>
# See the list of HTTP reponse codes for the full set!


#
# when exploring, you'll often obtain an unfamiliar object.
# Here, we'll ask what type it is
type(result)


# We can print all of the data members in an object with dir
# Since dir returns a list, we will grab that list and loop over it:
all_fields = dir(result)

for field in all_fields:
    if "_" not in field:
        print(field)


#
# Let's try printing a few of those fields (data members):
print(f"result.url         is {result.url}")  # the original url
print(f"result.raw         is {result.raw}")  # another object!
print(f"result.encoding    is {result.encoding}")  # utf-8 is very common
print(f"result.status_code is {result.status_code}")  # 200 is success!


# In this case, the result is a text file (HTML) Let's see it!
contents = result.text
print(contents)


# Yay!
# This shows that you are able to "scrape" an arbitrary HTML page...

# Now, we're off to more _structured_ data-gathering...


#
# we assign the url and obtain the api-call result into result
#    Note that result will be an object that contains many fields (not a simple string)
#

import requests

url = "http://api.open-notify.org/iss-now.json"   # this is sometimes called an "endpoint" ...
result = requests.get(url)

# if it succeeds, you should see <Response [200]>


#
# Let's try printing those shorter fields from before:
print(f"result.url         is {result.url}")  # the original url
print(f"result.raw         is {result.raw}")  # another object!
print(f"result.encoding    is {result.encoding}")  # utf-8 is very common
print(f"result.status_code is {result.status_code}")  # 200 is success!


#
# In this case, we know the result is a JSON file, and we can obtain it that way:
json_contents = result.json()
print(json_contents)

# Remember:  json_contents will be a _dictionary_


#
# Let's see how dictionaries work:

json_contents['message']

# thought experiment:  could we access the other components? What _types_ are they?!!


# JSON is a javascript dictionary format -- almost the same as a Python dictionary:
data = { 'key':'value',  'fave':42,  'list':[5,6,7,{'mascot':'Aliiien'}] }
print(data)


data['list']



data['list'][3]['mascot']



#
# here, we will obtain plain-text results from a request
# url = "https://www.cs.hmc.edu/~dodds/demo.html"  # try it + source
# url = "https://www.scrippscollege.edu/"          # another possible site... 403
# url = "https://www.pitzer.edu/"                  # another possible site... 200
# url = "https://www.cmc.edu/"                     # and another!             200
# url = "https://www.cgu.edu/"                       # Yay CGU!               403
result = requests.get(url)
print(f"result is {result}")        # Question: is the result a "200" response?!


#
# we assign the url and use requests.get to obtain the result into result_astro
#
#    Remember, result_astro will be an object that contains many fields (not a simple string)
#

import requests

url = "http://api.open-notify.org/astros.json"   # this is sometimes called an "endpoint" ...
result_astro = requests.get(url)
result_astro

# if it succeeded, you should see <Response [200]>


# If the request succeeded, we know the result is a JSON file, and we can obtain it that way.
# Let's call our dictionary something more specific:

astronauts = result_astro.json()
print(astronauts)

d = astronauts     # shorter to type

# Remember:  astronauts will be a _dictionary_

note = """ here's yesterday evening's result - it _should_ be the same this morning!

{"people": [{"craft": "ISS", "name": "Oleg Kononenko"}, {"craft": "ISS", "name": "Nikolai Chub"},
{"craft": "ISS", "name": "Tracy Caldwell Dyson"}, {"craft": "ISS", "name": "Matthew Dominick"},
{"craft": "ISS", "name": "Michael Barratt"}, {"craft": "ISS", "name": "Jeanette Epps"},
{"craft": "ISS", "name": "Alexander Grebenkin"}, {"craft": "ISS", "name": "Butch Wilmore"},
{"craft": "ISS", "name": "Sunita Williams"}, {"craft": "Tiangong", "name": "Li Guangsu"},
{"craft": "Tiangong", "name": "Li Cong"}, {"craft": "Tiangong", "name": "Ye Guangfu"}],
"number": 12, "message": "success"}
"""


# use this cell for the in-class challenges, which will be
#    (1) to extract the value 12 from the dictionary d
d['number']
#    (2) to extract the name "Sunita Williams" from the dictionary d
d['people'][8]['name']


# use this cell - based on the example above - to share your solutions to the Astronaut challenges...
d['people'][5]['name']
d['people'][0]['name'][:-3:-1]


#
# use this cell for your API call - and data-extraction
#
url = 'https://catfact.ninja/facts?page=31'
result_url = requests.get(url)
result_url
cat = result_url.json()
print('Why happy cats are good: "', cat['data'][9]['fact'], '"', sep='')


#
# use this cell for your webscraping call - optional data-extraction
#
url = 'https://tilde.pt/~fimdomeio/index2.html'
result_url = requests.get(url)
type(result_url)
contents = result_url.text
print(contents)


print('this is what happened when I clicked on it TwT:', contents[292:389])
print('amazing advice:', contents[2032:2083])



#
# throwing a single dart
#

import random

def dart():
    """Throws one dart between (-1,-1) and (1,1).
       Returns True if it lands in the unit circle; otherwise, False.
    """
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    print("(x,y) are", (x,y))   # you'll want to comment this out...

    if x**2 + y**2 < 1:
        return True  # HIT (within the unit circle)
    else:
        return False # missed (landed in one of the corners)

# try it!
result = dart()
print("result is", result)




# Try it ten times in a loop:

for i in range(10):
    result = dart()
    if result == True:
        print("   HIT the circle!")
    else:
        print("   missed...")


# try adding up the number of hits, the number of total throws
# remember that pi is approximately 4*hits/throws   (cool!)



#
# Write forpi(n)
#

#
# For the full explanation, see https://www.cs.hmc.edu/twiki/bin/view/CS5Fall2021/PiFromPieGold
#


# This is only a starting point
def forpi(N):
    """Throws N darts, estimating pi."""
    pi = 42     # A suitably poor initial estimate
    throws = 0  # No throws yet
    hits = 0    # No "hits" yet  (hits ~ in the circle)

    return hits

# Try it!
forpi(10)



#
# Write whilepi(n)
#

#
# For the full explanation, see https://www.cs.hmc.edu/twiki/bin/view/CS5Fall2021/PiFromPieGold
#


# This is only a starting point
def whilepi(err):
    """Throws N darts, estimating pi."""
    pi = 42     # A suitably poor initial estimate
    throws = 0  # No throws yet
    hits = 0    # No "hits" yet  (hits ~ in the circle)

    return throws


# Try it!
whilepi(.01)


#
# Your additional punctuation-style explorations (optional!)
#





