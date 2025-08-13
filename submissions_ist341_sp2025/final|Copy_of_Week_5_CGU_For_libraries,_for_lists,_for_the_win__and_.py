#
# Live demo
#

print("Start to guess...")

guess = 41

while True:   # just like an if... except it repeats!
    print("Not the right guess")

print("Guessing done!")


L = [ 'CGU', 'CMC', 'PIT', 'SCR', 'POM', 'HMC' ]
print("len(L) is", len(L))     # just for fun, try max and min of this L:  We win! (Why?!)


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

L = range(1, 101)
total = sum(L) #Sum all numbers
print("sum(L) is", total)




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
LC = [ vwl_once(c) for c in s ]
print("LC is", LC)
#vowel-patterns are [0, 1, 0, 1, 0]
s = "hello"
LC = [vwl_once(c) for c in s]
print("s is", s)
print("LC is", LC)

#vowel-patterns is[0, 0, 0, 1, 0, 0]
s = "strong"
LC = [vwl_once(c) for c in s]
print("s is", s)
print("LC is", LC)

#vowel-patterns [1, 0, 1, 1, 0, 0, 1, 0]
s = "education"
LC = [vwl_once(c) for c in s]
print("s is", s)
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


s = "Python"
print("scrabble_all('Python') should be 14 <->", scrabble_all(s))


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


A = [x**2 for x in range(10) if x % 2 == 0]
print("A is", A)



# Let's try thinking about these...

"""
A = [ n+2 for n in   range(40,42) ]
B = [ 42 for z in [0,1,2] ]
C = [ z for z in [42,42] ]
D = [ s[::-2] for s in ['fungic','y!a!y!'] ]
L = [ [len(w),w] for w in  ['Hi','IST'] ]
"""

# then, see if they work the way you predict...


#
# here, [1] create a function that scores punctuation as 1 and non-punctuation as 0, pun_one
#       [2] use the list-comprehension technique to create a punctuation-scoring function, pun_all

# use the above examples as a starting point!
def pun_one(c):
    """Returns 1 if c is a punctuation mark, otherwise 0"""
    if c in ".,!?;:-'\"()[]{}":
        return 1
    else:
        return 0

# Step 2: Define pun_all using List Comprehension
def pun_all(s):
    """Returns total punctuation count in s"""
    LC = [pun_one(c) for c in s]  # Apply pun_one on each character
    total = sum(LC)  # Sum up all punctuation counts
    return total




# tests
print("pun_all(PLANKTON) should be 4 <->", pun_all(PLANKTON))
print("pun_all(PATRICK) should be 4 <->", pun_all(PATRICK))
# They're more similar, punctuation-wise than scrabble-wise


#
# The two works of yours and two of another author's ("theirs")
#     These can - and should - be large
#

YOURS1 = """  Python is widely used for data science, web development,
and automation due to its simplicity and powerful libraries """

YOURS2 = """  Python's flexibility allows developers to build scalable
applications, automate tasks, and work with machine learning models """

THEIRS1 = """  Python is an interpreted language, making debugging easier
but sometimes slower than compiled languages."""

THEIRS2 = """  Python’s extensive libraries support networking,
file handling, and automation, making it highly versatile """


len(THEIRS2)


#
# Here, run your punctuation-comparisons (absolute counts)
print("pun_all(YOURS1) [Mohammed]:", pun_all(YOURS1))
print("pun_all(YOURS2) [IST341_Participant_5]:", pun_all(YOURS2))
print("pun_all(THEIRS1) [Other Author]:", pun_all(THEIRS1))
print("pun_all(THEIRS2) [Other Author]:", pun_all(THEIRS2))





#
# Here, run your punctuation-comparisons (relative, per-character counts)
print("YOURS1 punctuation density [Mohammed]:", pun_all(YOURS1) / len(YOURS1))
print("YOURS2 punctuation density [IST341_Participant_5]:", pun_all(YOURS2) / len(YOURS2))
print("THEIRS1 punctuation density [Other Author]:", pun_all(THEIRS1) / len(THEIRS1))
print("THEIRS2 punctuation density [Other Author]:", pun_all(THEIRS2) / len(THEIRS2))




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
print("max of L: ", max(L))
print("min of L: ", min(L))

count=0
for i in L:
  if i==92:
    count+=1
print("number of 92: ", count)

# try min and max, count of 42's, count of 92's, etc.



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
print(len(result))


sum([ 2, 3, 4 ]) / len([2,3,4])


LC = [ len(birthday_room()) for i in range(100) ]
print(LC)
sum(LC) / len(LC)



L = [ len(birthday_room()) for i in range(100000) ]
print("birthday room: L[0:5] are", L[0:5])
print("birthday room: L[-5:] are", L[-5:])
#
# Let's see what the average results were
print("len(L) is", len(L))
ave = sum(L)/len(L)
print("average is", ave )
print("max is", max(L))
# try min and max, count of 42's, count of 92's, etc.


[ x**2 for x in [3,5,7] ]


s = "ash"
s[2]


[  s[-1] for s in ["ash", "IST341_Participant_8", "mohammed"] ]


max(L)


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




# Here is a starting example:
L = [ rwalk(6) for i in range(1000) ]

# then, find the average of those 1000 experiments for rwalk(5)
average = sum(L)/len(L)
print("The average for radius==6 (for 1000 trials) was", average)





# Here is a starting example:
L = [ rwalk(7) for i in range(1000) ]

# then, find the average of those 1000 experiments for rwalk(5)
average = sum(L)/len(L)
print("The average for radius==7 (for 1000 trials) was", average)



# Here is a starting example:
L = [ rwalk(8) for i in range(1000) ]

# then, find the average of those 1000 experiments for rwalk(5)
average = sum(L)/len(L)
print("The average for radius==8 (for 1000 trials) was", average)



# Here is a starting example:
L = [ rwalk(9) for i in range(1000) ]

# then, find the average of those 1000 experiments for rwalk(5)
average = sum(L)/len(L)
print("The average for radius==9 (for 1000 trials) was", average)



# Here is a starting example:
L = [ rwalk(10) for i in range(1000) ]

# then, find the average of those 1000 experiments for rwalk(5)
average = sum(L)/len(L)
print("The average for radius==10 (for 1000 trials) was", average)



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
#url = "https://www.cgu.edu/"
url = "https://www.facebook.com/terms?section_id=section_3"
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


# http://api.open-notify.org/iss-now.json



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
num_astronauts = d["number"]
print("Total astronauts:", num_astronauts)
#    (2) to extract the name "Sunita Williams" from the dictionary d
for person in d["people"]:
    if person["name"] == "Sunita Williams":
        print("Found:", person["name"])


# use this cell - based on the example above - to share your solutions to the Astronaut challenges...
for person in d["people"]: #Find "Jeanette Epps"
    if person["name"] == "Jeanette Epps":
        print("Found:", person["name"])

for person in d["people"]: #Find ok using "Nikolai Chub"
    if person["name"] == "Nikolai Chub":
        print("Status:", "ok")



#
# use this cell for your API call - and data-extraction
import requests

url = "https://api.open-meteo.com/v1/forecast?latitude=24.7136&longitude=46.6753&current_weather=true" # API for Riyadh, Saudi Arabia

result_weather = requests.get(url) #API request


print(f"API Response: {result_weather.status_code}")

weather_data = result_weather.json() #Extract and print the JSON data
print(weather_data)

temperature = weather_data["current_weather"]["temperature"]
wind_speed = weather_data["current_weather"]["windspeed"]


print(f"Current Temperature in Riyadh: {temperature}°C")
print(f"Wind Speed in Riyadh: {wind_speed} km/h") #Print extracted values



#
# use this cell for your webscraping call - optional data-extraction
import requests

url = "https://en.wikipedia.org/wiki/Main_Page" #webpage URL (Wikipedia Main Page)
result_html = requests.get(url) #To get the HTML content
print(f"Webpage Response: {result_html.status_code}") #Print the status to ensure success

html_content = result_html.text[:500]  # Extracting first 500 characters
print(html_content)




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
    #print("(x,y) are", (x,y))   # you'll want to comment this out...

    if x**2 + y**2 < 1:
        return True  # HIT (within the unit circle)
    else:
        return False # missed (landed in one of the corners)

# try it!
result = dart()
print("result is", result)




# Try it ten times in a loop:
hitnumber=0
totalthrows=0
for i in range(10):
    result = dart()
    totalthrows+=1
    if result == True:
        hitnumber+=1
        print("   HIT the circle!")
    else:
        print("   missed...")
pi = 4*hitnumber/totalthrows
print("The number of hits: ", hitnumber)
print("The number of throws: ", totalthrows)
print("The pi is: ", pi )

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
    for i in range(N):
        throws+=1
        if dart():
            hits+=1
        pi= 4*hits/throws
        print ('This is hit number: ', hits, 'out of ', throws,'  throws, so Pi is :', pi)

# Try it!
forpi(10)



#
# Write whilepi(n)
#

#
# For the full explanation, see https://www.cs.hmc.edu/twiki/bin/view/CS5Fall2021/PiFromPieGold
#


# This is only a starting point
import math
def whilepi(error):
    """Throws N darts, estimating pi."""
    pi = 42     # A suitably poor initial estimate
    throws = 0  # No throws yet
    hits = 0    # No "hits" yet  (hits ~ in the circle)
    while abs ((math.pi) - pi) >= error:
        throws+=1
        if dart():
            hits+=1
        pi= 4*hits/throws
        print ('This is hit number: ', hits, 'out of ', throws,'  throws, so Pi is :', pi)

    return throws


# Try it!
whilepi(.01)


#
# Your additional punctuation-style explorations (optional!)
#





