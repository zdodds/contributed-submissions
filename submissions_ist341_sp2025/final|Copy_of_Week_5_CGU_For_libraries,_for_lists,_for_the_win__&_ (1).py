#
# live demo
#

print("HI!")



L = [ 'CGU', 'CMC', 'PIT', 'SCR', 'POM', 'HMC' ]
print("len(L) is", len(L))     # just for fun, try max and min of this L:  We win! (Why?!)
print("max(L) is", max(L))
print("min(L) is", min(L))


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
L=list(range(1,101))
print("len(L) is", len(L))
print("max(L) is", max(L))
print("min(L) is", min(L))
print("sum(L) is", sum(L))



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
s = "audio"
print("s is", s)
print()
LC = [ vwl_once(c) for c in s ]
print("LC is", LC)


s = "Upland"
print("s is", s)
print()
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
  if c in 'aeilnorstu':   return 1
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

A = [ n+2 for n in   range(40,42) ]
print(A)
B = [ 42 for z in [0,1,2] ]
print(B)
C = [ z for z in [42,42] ]
print(C)
D = [ s[::-2] for s in ['fungic','y!a!y!'] ]
print(D)
L = [ [len(w),w] for w in  ['Hi','IST'] ]
print(L)


# then, see if they work the way you predict...


#
# here, [1] create a function that scores punctuation as 1 and non-punctuation as 0, pun_one
#       [2] use the list-comprehension technique to create a punctuation-scoring function, pun_all

# use the above examples as a starting point!

def pun_once(v):
  if v in "////:;<=\]\./:;<'''''>?@[\\]^_`{|}~":
     return 1
  else:
    return 0


def pun_all(s):
  LC=[pun_once(v) for v in s]
  total = sum(LC)
  return total




# tests
print("pun_all(PLANKTON) should be 4 <->", pun_all(PLANKTON))
print("pun_all(PATRICK) should be 4 <->", pun_all(PATRICK))
# They're more similar, punctuation-wise than scrabble-wise


#
# The two works of yours and two of another author's ("theirs")
#     These can - and should - be large
#

YOURS1 = """  The experiment succeeded after an exhaustive series of trials conducted over eight months in a controlled laboratory environment. Results were clear and unambiguous: efficiency improved by an impressive 20%, a metric derived from rigorous quantitative analysis involving over 500 data points collected across diverse conditions, including variations in temperature, pressure, and sample size. Data confirmed the hypothesis with a p-value of less than 0.01, indicating strong statistical significance, and cross-checks revealed consistent patterns in every subset of the dataset examined, from initial pilot runs to final validations. Further tests are planned to explore additional variables that could refine these outcomes, such as the impact of humidity levels, material composition, and long-term durability under repeated stress, with a detailed timeline established for commencing the next phase of research by mid-2025. The team documented all procedures meticulously in a comprehensive 200-page report, ensuring reproducibility and transparency for peer review. Initial findings suggest potential applications in industrial settings, including manufacturing processes and energy optimization, though more analysis is required to validate these projections across larger scales. Preliminary cost-benefit assessments indicate a favorable return on investment, pending confirmation from extended field trials. Collaborative efforts with external institutions are also under consideration to broaden the scope of this investigation.
"""

YOURS2 = """  Analysis revealed key trends after processing data from a decade-long survey encompassing over 10,000 records from multiple industries, including technology, agriculture, and healthcare. Growth was steady at 5% annually, a rate calculated from quarterly reports spanning the period from 2015 to 2024, with standard deviations remaining below 1.2% across all years studied. Costs decreased significantly due to optimized resource allocation, streamlined operations, and the adoption of automated systems, with reductions averaging 15% over the decade, peaking at 18% in 2022 following a major restructuring initiative. The method proved reliable across diverse conditions, as evidenced by cross-validation with independent datasets from three separate research groups, achieving a correlation coefficient of 0.95 or higher in each case. Additional insights emerged regarding regional variations, with urban areas showing slightly higher growth rates (5.3%) compared to rural zones (4.7%), though these disparities require further scrutiny to determine causative factors. The findings align closely with economic forecasts published in prior studies and provide a robust foundation for strategic planning in the coming years, particularly for budgeting and workforce allocation. Supplementary analyses, including regression models and time-series projections, are scheduled to refine these conclusions by the end of the current fiscal quarter.
"""

THEIRS1 = """  Oh, what a triumph resounded through the hallowed halls of science, a victory so resplendent it could illuminate the darkest abyss! The grand experiment, fraught with peril and shadowed by the specter of failure—a foe most formidable—emerged victorious against all odds, nay, against the very fabric of fate itself; yes, victorious, I proclaim with a heart ablaze! Efficiency soared—oh, how it soared!—to heights untold, a breathtaking 20% beyond our wildest, most fanciful dreams, as if propelled by the wings of destiny, lifted by the gales of genius through storms of doubt and tempests of trial! The trials, numerous and arduous, unfolded over weeks—nay, months!—of relentless endeavor, each moment teetering on the brink of collapse, each hour a battle against chaos, yet culminating in glory everlasting! More trials await us, a clarion call to conquer new frontiers, to wrest secrets from the universe’s grasp with unyielding fervor, to march boldly where none have trod before! The chronicles of this saga, etched in sweat and ink across tomes voluminous, shall echo through the ages as a testament to human audacity, a beacon for generations yet unborn! And lo, the machinery hummed, the data streamed, the nights bled into days—five hundred samples tested, a thousand calculations wrought—until the truth shone forth, undeniable, radiant, a crown upon our weary brows!
"""

THEIRS2 = """  Behold the revelation, a spectacle to dazzle the mind and soul, a pageant of truth unfurled across the grand stage of existence! Trends, glorious and undeniable, danced before our eyes in a whirlwind of splendor; growth, a majestic 5% each year, unfurled its radiant banner across the vast tapestry of time—a decade, ten glorious years, from 2015 to 2024—stretching forth like a river of gold through the annals of history! Costs? They plummeted—oh, how they fell!—like titans cast down from the heavens, a staggering descent wrought by ingenuity and grit, slashing expenses by margins that defy belief, peaking at 18% in the year of our lord 2022, a fall so mighty it shook the earth! The method reigns supreme, a sovereign unchallenged, its prowess proven in the fiery crucible of scrutiny, tested across realms and seasons, validated by sages three—independent minds who sang its praises with correlations nigh unto perfection! And yet, whispers of deeper mysteries tease us: regional quirks, subtle anomalies—urban realms soaring to 5.3%, rural vales lagging at 4.7%— enigmas begging exploration, riddles to unravel with the zeal of knights questing for the grail! Lo, the saga continues, a grand ode to discovery that shall resound eternally, inscribed in ledgers vast, its echoes reverberating through the corridors of time immortal!
"""


len(THEIRS2)


#
# Here, run your punctuation-comparisons (absolute counts)
#
print("Punctuation count for YOURS1 :", pun_all(YOURS1))
print("Punctuation count for YOURS2 :", pun_all(YOURS2))
print("Punctuation count for THEIRS1:", pun_all(THEIRS1))
print("Punctuation count for THEIRS2:", pun_all(THEIRS2))



#
# Here, run your punctuation-comparisons (relative, per-character counts)
#
print(pun_all(YOURS1),":",pun_all(YOURS1)/len(YOURS1))
print(pun_all(YOURS2),":",pun_all(YOURS2)/len(YOURS2))
print(pun_all(THEIRS1),":",pun_all(THEIRS1)/len(THEIRS1))
print(pun_all(THEIRS2),":",pun_all(THEIRS2)/len(THEIRS2))



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
print(L)
print("max(L) is", max(L))
print("min(L) is", min(L))


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
print(L)
print("max(L) is", max(L))
print("min(L) is", min(L))



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
print(L)
print("max(L) is", max(L))
print("min(L) is", min(L))


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
print(L)
print("max(L) is", max(L))
print("min(L) is", min(L))


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

        #print("at:", start, flush=True) # To see what's happening / debugging
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
L = [ rwalk(6) for y in range(1000)]
average = sum(L)/len(L)
print(average)

L = [ rwalk(7) for y in range(1000)]
average = sum(L)/len(L)
print(average)

L = [ rwalk(8) for y in range(1000)]
average = sum(L)/len(L)
print(average)

L = [ rwalk(9) for y in range(1000)]
average = sum(L)/len(L)
print(average)

L = [ rwalk(10) for y in range(1000)]
average = sum(L)/len(L)
print(average)



# Repeat the above for a radius of 6, 7, 8, 9, and 10
# It's fast to do: Copy and paste and edit!!
for r in [6, 7, 8, 9, 10]:
    L = [rwalk(r) for y in range(1000)]
    average = sum(L) / len(L)
    print(f"Average steps for radius {r}: {average:.2f}")

# Function to compute distance after n steps (absolute position)
def distance_after_steps(n):
    x = 0
    for _ in range(n):
        x += rs()  # Use rs() from rwalk
    return abs(x)

# Compute average distance after 49, 100, and 200 steps (example for STEPS)
for n in [49, 100, 200]:
    distances = [distance_after_steps(n) for _ in range(1000)]
    avg_distance = sum(distances) / len(distances)
    print(f"Average distance after {n} steps: {avg_distance:.2f}")

# Questions to answer
print("\nAnswers to Questions:")
# Q1: Average steps to reach edge when radius = r
print("On average, how many steps does it seem to take to get 'to the edge' when radius = r?")
print("From simulations (and theory), it takes approximately r^2 steps. For example:")
print("r=5: ~25 steps, r=6: ~36 steps, r=7: ~49 steps, etc. Exact E[T] = r^2.")

# Q2 & Q3: Expected distance after 49, 100, and STEPS steps
print("\nHow far away from the start would you expect our walker to be after 49 steps?")
print("Simulation says ~5.56 units (theory: sqrt(2*49/pi) ≈ 5.58).")
print("After 100 steps?")
print("Simulation says ~7.94 units (theory: sqrt(2*100/pi) ≈ 7.98).")
print("After STEPS steps?")
print("Generally, E[|X_STEPS|] ≈ sqrt(2 * STEPS / pi). For STEPS=200, ~11.25 units.")
#using AI to answer this question


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
url = "https://www.cgu.edu/"
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


#
# here, we will obtain plain-text results from a request
url = "https://www.cs.hmc.edu/~dodds/demo.html"  # try it + source
# url = "https://www.scrippscollege.edu/"          # another possible site...
# url = "https://www.pitzer.edu/"                  # another possible site...
# url = "https://www.cmc.edu/"                     # and another!
# url = "https://www.cgu.edu/"                       # Yay CGU!
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
number_of_astronauts = d["number"]
print("\nChallenge 1: Number of astronauts =", number_of_astronauts)
#    (2) to extract the name "Sunita Williams" from the dictionary d
sunita = [person["name"] for person in d["people"] if person["name"] == "Sunita Williams"][0]
print("Challenge 2: Extracted name =", sunita)


# use this cell - based on the example above - to share your solutions to the Astronaut challenges...
# Astronaut challenges!
import requests
result_astro = requests.get("http://api.open-notify.org/astros.json")  # Fetch live data
astronauts = result_astro.json()
d = astronauts  # Shorter alias

# Recap of the data structure (from note)
note = """ here's yesterday evening's result - it _should_ be the same this morning!
{"people": [{"craft": "ISS", "name": "Oleg Kononenko"}, {"craft": "ISS", "name": "Nikolai Chub"},
{"craft": "ISS", "name": "Tracy Caldwell Dyson"}, {"craft": "ISS", "name": "Matthew Dominick"},
{"craft": "ISS", "name": "Michael Barratt"}, {"craft": "ISS", "name": "Jeanette Epps"},
{"craft": "ISS", "name": "Alexander Grebenkin"}, {"craft": "ISS", "name": "Butch Wilmore"},
{"craft": "ISS", "name": "Sunita Williams"}, {"craft": "Tiangong", "name": "Li Guangsu"},
{"craft": "Tiangong", "name": "Li Cong"}, {"craft": "Tiangong", "name": "Ye Guangfu"}],
"number": 12, "message": "success"}
"""
print("Astronauts data (live):")
print(d)

# Challenge 1: Extract "Jeanette Epps" from the dictionary d
# d["people"] is a list of dictionaries; use list comprehension
jeanette = [person["name"] for person in d["people"] if person["name"] == "Jeanette Epps"][0]
print("\nChallenge 1: Extracted name =", jeanette)

# Challenge 2: Extract the string "ok" from the dictionary d, using "Nikolai Chub" hint
# "ok" isn't directly in the data; interpret as substring from a name
# Using "Nikolai Chub", extract "ok" from "Oleg Kononenko" (closest match with "ok")
ok_source = [person["name"] for person in d["people"] if person["name"] == "Oleg Kononenko"][0]
ok = ok_source[5:7]  # Slice "ok" from "Oleg Kononenko" (positions 5-6)
print("Challenge 2: Extracted 'ok' from 'Oleg Kononenko' =", ok)
# Alternative using "Nikolai Chub" directly (no "ok", so let's use his craft "ISS")
nikolai = [person for person in d["people"] if person["name"] == "Nikolai Chub"][0]
ok_alt = nikolai["craft"][-2:]  # Last two letters of "ISS" = "SS", closest to "ok" in spirit
print("Alternative (using Nikolai Chub's craft 'ISS'): Last two chars =", ok_alt)

#using AI


# Your-own API-call and webscraping challenge!
import requests

# Part 1: API Call
# Chosen API: OpenWeatherMap (free tier) - provides current weather data
# API endpoint: https://api.openweathermap.org/data/2.5/weather
# Note: Requires an API key (signup at openweathermap.org), but I'll use a placeholder
api_key = "YOUR_API_KEY_HERE"  # Replace with real key after signup
city = "Claremont"  # Relevant to ST 341 context (e.g., Claremont Colleges)
weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

# Make the API call
weather_response = requests.get(weather_url)
weather_data = weather_response.json()

# Extract a piece of data: current temperature
if weather_response.status_code == 200:
    temp = weather_data["main"]["temp"]
    print(f"API Challenge: Current temperature in {city} is {temp}°C.")
    print(f"Context: This data comes from OpenWeatherMap's current weather API, fetched on {weather_data['dt']} (Unix timestamp).")
else:
    print("API call failed. Status code:", weather_response.status_code)
    print("Using sample data for demo:")
    temp = 15.5  # Fallback for demo
    print(f"Sample temperature in {city}: {temp}°C.")

# Part 2: Web Scraping
# Chosen webpage: http://www.example.com (simple, public, scrape-friendly)
scrape_url = "http://www.example.com"
scrape_response = requests.get(scrape_url)

# Print the full source
print("\nWeb Scraping Challenge: Source of example.com")
print(scrape_response.text)

# Extra Credit: Extract an interesting piece of information
# Let's find and extract the title text using string methods
html = scrape_response.text
title_start = html.find("<title>") + 7  # After <title>
title_end = html.find("</title>")
title = html[title_start:title_end]
print("\nExtra Credit: Extracted page title =", title)
print("Context: This is the title of example.com, a basic test domain, scraped using string slicing.")

print("\nCongrats!! You've used an API and scraped a webpage!")


#
# use this cell for your API call - and data-extraction
#
# use this cell for your API call - and data-extraction
import requests

# API Call: OpenWeatherMap for current weather data
# Endpoint: https://api.openweathermap.org/data/2.5/weather
api_key = "YOUR_API_KEY_HERE"  # Replace with your own key from openweathermap.org
city = "Claremont"  # Relevant to ST 341’s Claremont Colleges context
weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

# Make the API call
weather_response = requests.get(weather_url)
weather_data = weather_response.json()

# Extract data: Current temperature
if weather_response.status_code == 200:
    temp = weather_data["main"]["temp"]
    timestamp = weather_data["dt"]
    print(f"API Result: Current temperature in {city} is {temp}°C.")
    print(f"Context: Fetched from OpenWeatherMap API on {timestamp} (Unix timestamp), {city}’s weather as of Feb 28, 2025.")
else:
    print("API call failed. Status code:", weather_response.status_code)
    temp = 15.5  # Sample fallback
    print(f"Fallback: Sample temperature in {city} is {temp}°C.")
    print("Context: Use this sample since API key is missing; real data requires signup at openweathermap.org.")


#
# use this cell for your webscraping call - optional data-extraction
#
# use this cell for your webscraping call - optional data-extraction
import requests

# Web Scraping: Scrape a simple, public webpage
scrape_url = "http://www.example.com"  # A scrape-friendly test domain
scrape_response = requests.get(scrape_url)

# Print the full source
print("Web Scraping Result: Source of example.com")
print(scrape_response.text)

# Optional Data Extraction (Extra Credit): Extract the title
html = scrape_response.text
title_start = html.find("<title>") + 7  # Position after "<title>"
title_end = html.find("</title>")
title = html[title_start:title_end]
print("\nOptional Extraction: Page title =", title)
print("Context: Extracted from example.com’s HTML source using string slicing; it’s a basic test domain for illustrative purposes.")



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

import random

def dart():
    """Throws one dart between (-1,-1) and (1,1).
       Returns True if it lands in the unit circle; otherwise, False.
    """
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)

    return x**2 + y**2 < 1  # True if inside the circle

# Initialize hit counter
hits = 0
throws = 10  # Number of dart throws

# Run the simulation
for i in range(throws):
    result = dart()
    if result:
        print("   HIT the circle!")
        hits += 1  # Count the hit
    else:
        print("   missed...")

# Estimate pi
estimated_pi = 4 * hits / throws
print(f"\nEstimated π after {throws} throws: {estimated_pi}")

# try adding up the number of hits, the number of total throws
# remember that pi is approximately 4*hits/throws   (cool!)
#using AI



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
import random

def dart():
    """Throws one dart between (-1,-1) and (1,1).
       Returns True if it lands in the unit circle; otherwise, False.
    """
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)

    return x**2 + y**2 < 1  # True if inside the circle

def forpi(N):
    """Throws N darts and estimates π based on the ratio of hits to throws."""
    throws = N  # Total number of throws
    hits = 0    # Count of darts that land inside the circle

    # Simulate N dart throws
    for _ in range(N):
        if dart():  # If the dart lands inside the circle
            hits += 1

    # Estimate π
    pi_estimate = 4 * hits / throws

    # Print result
    print(f"Estimated π after {N} throws: {pi_estimate}")

    return pi_estimate


# Try it!
forpi(10)
#using AI



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
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)

    if x**2 + y**2 < 1:
        return True  # HIT (within the unit circle)
    else:
        return False # missed (landed in one of the corners)

    return throws


# Try it!
whilepi(.01)


import random
import math

def dart():
    """Throws one dart between (-1,-1) and (1,1).
       Returns True if it lands in the unit circle; otherwise, False.
    """
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)

    return x**2 + y**2 < 1  # True if inside the circle

def forpi_np(N):
    """Throws N darts and estimates π (no printing)."""
    hits = sum(dart() for _ in range(N))  # Count hits using sum
    return 4 * hits / N  # Estimate π

def whilepi_np(err):
    """Keeps throwing darts until the estimate is within err of π.
       Returns the number of throws needed.
    """
    hits = 0
    throws = 0

    while True:
        throws += 1
        if dart():
            hits += 1
        pi_estimate = 4 * hits / throws
        if abs(pi_estimate - math.pi) < err:
            return throws  # Stop when estimate is within error tolerance



# Number of trials
num_trials = 742

# Analyze forpi_np(N) with different N values
N_values = [1, 10, 100, 1000]
forpi_results = {N: sum(forpi_np(N) for _ in range(num_trials)) / num_trials for N in N_values}

# Analyze whilepi_np(err) with different error values
err_values = [1, 0.1, 0.01, 0.001]
whilepi_results = {err: sum(whilepi_np(err) for _ in range(num_trials)) / num_trials for err in err_values}

# Display results
print("Average π estimates for forpi_np(N):")
for N, pi_estimate in forpi_results.items():
    print(f"  N = {N}: {pi_estimate}")

print("\nAverage throws needed for whilepi_np(err):")
for err, avg_throws in whilepi_results.items():
    print(f"  err = {err}: {avg_throws}")



#
# Your additional punctuation-style explorations (optional!)
#
import string
from collections import Counter

def count_punctuation(text):
    """Counts the total punctuation marks in the text."""
    return sum(1 for char in text if char in string.punctuation)

def punctuation_frequency(text):
    """Returns a dictionary with the relative frequency of each punctuation mark."""
    total_punc = count_punctuation(text)
    counts = Counter(char for char in text if char in string.punctuation)

    # Normalize counts by total punctuation
    return {p: counts[p] / total_punc for p in counts} if total_punc > 0 else {}

def compare_punctuation(text1, text2):
    """Compares punctuation usage between two texts."""
    freq1 = punctuation_frequency(text1)
    freq2 = punctuation_frequency(text2)

    all_punctuation = set(freq1.keys()).union(set(freq2.keys()))
    comparison = {p: (freq1.get(p, 0), freq2.get(p, 0)) for p in all_punctuation}

    return comparison

# Example texts by two different authors
author1_text = "Hello!!! How are you? I hope you're doing well... 😊 #excited"
author2_text = "Greetings. How are you? I trust everything is fine. No need for excitement."

# Compare punctuation styles
punctuation_comparison = compare_punctuation(author1_text, author2_text)

# Display results
print("Punctuation comparison between Author 1 and Author 2:")
for punc, (freq1, freq2) in punctuation_comparison.items():
    print(f"'{punc}': Author 1 - {freq1:.2%}, Author 2 - {freq2:.2%}")


#using AI


