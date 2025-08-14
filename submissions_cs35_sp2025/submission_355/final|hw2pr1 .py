#
# computing-styling trick of the day     (or, of the minute...)
#
# The setting for word-wrapping on the output is
#     "notebook.output.wordWrap": true,   (in your settings.json file or from Code ... Settings ...) 

print( list(range(100)) )




#
# see if we have the requests library...
#

import requests


#
# If you _don't_ have the requests library, let's install it!
#

# for me, it worked to uncomment and run this command, here in this cell:
# #3 install requests  OR   # install requests

# an alternative is to run, in a terminal, the command would be 
#  #3 install requests  OR    # install requests      (the ! is needed only if inside Python)

# It's very system-dependent how much you have to "restart" in order to use
# the new library (the notebook, VSCode, the Jupyter extension, etc.)

# Troubles?  Let us know!  We'll work on it with you...


#
# hopefully, this now works! (if so, running will succeed silently)
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
result    

# if it succeeded, you should see <Response [200]>
# See the list of HTTP reponse codes for the full set!


#
# when exploring, you'll often obtain an unfamiliar object. 
# Here, we'll ask what type it is 
type(result)


# here is one of the data members within the result
# it "remembers" (keeps track of) the url requested:
result.headers


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
result    

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
# Let's re-remind ourselves how dictionaries work:

CS35_Participant_2 = float(json_contents['iss_position']['longitude']) 

CS35_Participant_2     # Challenge:  could we access the other components? What _types_ are they?!!


#
# In Python, we can use the resulting dictionary... let's see its keys:
print(list(json_contents.keys()))  

# Also, watch out for string vs. numeric types, e.g., for latitude and longitude.
# At heart, _all_ web data are strings... .

# These experiments will be helpful for problem 1, below :)


# JSON is a javascript dictionary format -- almost the same as a Python dictionary:
data = { 'key':'value',  'fave':42,  'list':[5,6,7,{'mascot':'Aliiien'}] }
print(data)

# we can write in JSON format to a local file, named small42.json:
import json 

with open("small.json", "w") as f:
    json.dump( data, f )


# We can also read from a json file
# The resulting data will be a _dictionary_:

with open("small.json", "r") as f:
    dictionary = json.load( f )

print(f"the {dictionary = }")


# let's access this dictionary -- first, the keys:
list(dictionary.keys())   # How do we get 'Aliiien' from newdata?


# Task: use the dictionary to obtain (a) 'value' , (b) 42 , (c) 'Aliiien'  [tricky!]

# remember that there are two ways to get the value from a key:
# way 1:  dictionary['key']            # errors if 'key' isn't present
# way 2:  dictionary.get('key')        # returns None if 'key' isn't present

dictionary['key'] #a
dictionary['fave'] #b
dictionary['list'][3]['mascot'] #c


import requests 

# here, we will obtain plain-text results from a request
url = "https://www.cs.hmc.edu/~dodds/demo.html"  # try it + source
# url = "https://www.scrippscollege.edu/"          # another possible site...
# url = "https://www.pitzer.edu/"                  # another possible site...
# url = "https://www.cmc.edu/"                     # and another!
# url = "https://www.cgu.edu/"
result = requests.get(url)        
print(f"result is {result}")        # hopefully it's 200


# if the request was successful, the Response will be [200]. 
# Then, we can grab the text - or json - from the site:

text = result.text                  # provides the HTML page as a large string...
print(f"len(text) is {len(text)}")  # let's see how large the HTML page is... 

print("\nThe first 242 characters are\n")
print(text[:])                  # we'll print the first few characters...  

# change this to text[:] to see the whole document...
# Notice that we can run many different analyses without having to re-call/re-scrape the page (this is good!)


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
d = astronauts     # d is shorter to type

# Remember:  d and astronauts will be a _dictionary_

note = """ here's yesterday's result - it _should_ be the same today!

{"people": [{"craft": "ISS", "name": "Oleg Kononenko"}, {"craft": "ISS", "name": "Nikolai Chub"},
{"craft": "ISS", "name": "Tracy Caldwell Dyson"}, {"craft": "ISS", "name": "Matthew Dominick"},
{"craft": "ISS", "name": "Michael Barratt"}, {"craft": "ISS", "name": "Jeanette Epps"},
{"craft": "ISS", "name": "Alexander Grebenkin"}, {"craft": "ISS", "name": "Butch Wilmore"},
{"craft": "ISS", "name": "Sunita Williams"}, {"craft": "Tiangong", "name": "Li Guangsu"},
{"craft": "Tiangong", "name": "Li Cong"}, {"craft": "Tiangong", "name": "Ye Guangfu"}], "number": 12, "message": "success"}
"""
""

len(d['people'])

name = 'Michael Barratt'
name[-12::-2]


#
# Cell to try out parsing d  (astronauts)
#




#
# Let's try the  count  endpoint, with geojson format (json with geographical data)
#

url = "https://earthquake.usgs.gov/fdsnws/event/1/count?format=geojson&minmagnitude=5.0&starttime=2024-01-01&endtime=2024-02-01"

result = requests.get(url)                       # a named input, params, taking the value param_d, above
print(f"result is {result}")                     # hopefully, this is 200
print(f"the full url used was\n {result.url}")   # it's nice to be able to see this


# If it worked, we should be able to obtain the JSON. Remember, it's a dictionary. Let's use d:

d = result.json()

print(f"{d =}")


#
# Here is the endpoint
#
url = "https://earthquake.usgs.gov/fdsnws/event/1/count"

# Let's use variables for three of the parameters:
min_mag = 5.0               # the minimum magnitude considered a quake (min_mag)
start_time = "2025-01-01"   # this is the year-month-day format of the start
finish_time = "2025-02-01"  # similar for the end

# we assemble a dictionary of our parameters, let's name it param_dictionary
# there are many more parameters available. The problems below ask you to explore them...
param_dictionary = { "format":"geojson",         # this is simply hard-coded to obtain json
                     "starttime":start_time,
                     "endtime":finish_time,
                     "minmagnitude":min_mag,
                     }

# Here, we use requests to make the request. The parameters will be added by this API call:
result = requests.get(url, params=param_dictionary)
print(f"result is {result}")                     # hopefully, this is 200
print(f"the full url used was\n {result.url}")   # this will include the parameters!


# If it worked, we should be able to see the json results:

d = result.json()
print(f"JSON returned was {d = }")


#
# How many quakes of magnitude >= 4.2 have been within 300km of Claremont 
#     + in Jan 2025
#     + in Dec 2024
#
url = "https://earthquake.usgs.gov/fdsnws/event/1/count"

# Let's use variables for three of the parameters:
min_mag = 2.2               # the minimum magnitude considered a quake (min_mag)
start_time = "2025-01-01"   # this is the year-month-day format of the start
finish_time = "2025-02-01"  # similar for the end
# start_time = "2024-01-01"   # similar, but for a year-CS35_Participant_2 span...
# finish_time = "2025-01-01"  # similar for the end
radius_in_km = 300

# we assemble a dictionary of our parameters, let's name it param_dictionary
# there are many more parameters available. The problems below ask you to explore them...
param_dictionary = { "format":"geojson",         # this is simply hard-coded to obtain json
                     "starttime":start_time,
                     "endtime":finish_time,
                     "minmagnitude":min_mag,
                     "latitude":34.0967,
                     "longitude":-117.7198,
                     "maxradiuskm":radius_in_km,
                     }

# Here, we use requests to make the request. The parameters will be added by this API call:
result = requests.get(url, params=param_dictionary)
print(f"result is {result}")                     # hopefully, this is 200
print(f"the full url used was\n {result.url}")   # this will include the parameters!

# We'll extract the final result in another cell:


# Let's finish up here:
quake_count = result.json()
print(f"{quake_count = }")


#
# Here is the endpoint
#
url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Let's use variables for three of the parameters:
min_mag = 7.4                 # the minimum magnitude considered a quake (min_mag)
start_time = "2024-02-01"   # this is the year-month-day format of the start
finish_time = "2025-02-01"  # similar for the end

# we assemble a dictionary of our parameters, let's name it param_dictionary
# there are many more parameters available. The problems below ask you to explore them...
param_dictionary = { "format":"geojson",         # this is simply hard-coded to obtain json
                     "starttime":start_time,
                     "endtime":finish_time,
                     "minmagnitude":min_mag,
                     }

# Here, we use requests to make the request. The parameters will be added by this API call:
result = requests.get(url, params=param_dictionary)
print(f"result is {result}")                     # hopefully, this is 200
print(f"the full url used was\n {result.url}")   # this will include the parameters!


# If it worked, we should be able to see the json results:

d = result.json()
print(f"JSON returned was {d = }")


#
# That's hard to read!
# Let's pretty-print it with the json library
#       Also, this version can be pasted into online formatters, e.g., https://jsonformatter.org/

import json 
nice_string = json.dumps(d)   # this outputs a "nicely formatted string" using double quotes
print(nice_string)




import json 
nicer_string = json.dumps(d, indent=4)   # We can specify the indentation. 
print(nicer_string)

                   





import datetime
import pytz  # Make sure you have pytz installed

# Convert the timestamp to seconds (from milliseconds)
timestamp = 1721353848571 / 1000

# Convert to a readable date and time format using UTC
readable_time = datetime.datetime.fromtimestamp(timestamp, tz=pytz.UTC)
print(readable_time)



#
# hw2: ISS tasks 1 and 2 ...
# 
# Two functions:  ISS_now(), ISS_distance()

#
# Use the ISS examples above to write a function, named 
#     
#      ISS_now()
#
# that uses requests to return the current latitude and longitude -- as floating-point values -- right now.
# Be sure to test it! 

import requests
d = {}


def ISS_now(url):
    """takes a url input, returns floats of the ISS' current latitude and longitude"""

    result_loc = requests.get(url)
    coord = result_loc.json()
    d = coord

    loc = d['iss_position']
    lat = loc['latitude']
    lat = float(lat)
    CS35_Participant_2 = loc['longitude']
    CS35_Participant_2 = float(CS35_Participant_2)

    return lat, CS35_Participant_2


url = "http://api.open-notify.org/iss-now.json"
current = ISS_now(url)

print(f"The current location of the ISS is at latitude {current[0]} and longitude {current[1]}.")


# ISS_now tests

import time

url = "http://api.open-notify.org/iss-now.json"
current = ISS_now(url)

time.sleep(3) # pause for 3 sec

new_current = ISS_now(url)

assert type(current[0]) == float
assert type(current[1]) == float

assert new_current[0] != current[0] # checking that location is updating


# 
# Once your ISS_now() function is working, write a new function
#
#       ISS_distance()
#
# which uses ISS_now to obtain the lat/CS35_Participant_2 of the ISS and then
# uses the haversine distance (look up a Python implementation or use one of ours... :)
# to compute the ISS's distance from a city of your choice.
#
# The haversine distance computes the "great circle" distance from two points on the globe
#     using latitude and longitude  
#

# city of choice = Chicago
Chi_lat = 41.8781 # north
Chi_long = -87.6298 # west
Chicago = [Chi_lat, Chi_long]

url = "http://api.open-notify.org/iss-now.json"


from math import *

def haversine(lat1, long1, lat2, long2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    long1, lat1, long2, lat2 = map(radians, [long1, lat1, long2, lat2])


    # haversine formula
    dlong = long2 - long1
    dlat = lat2 - lat1
    trig = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlong/2)**2
    # Radius of earth. Use 3956 for miles. 6371 for km.
    radius = 3956  # we'll use miles!
    return radius * 2 * asin(sqrt(trig))

def ISS_distance(city):
    """take a city input, which should be a list of the city's lat and CS35_Participant_2, and returns a float of the current distance between that city and the ISS"""

    ISS_loc = ISS_now(url)
    ISS_lat = ISS_loc[0]
    ISS_long = ISS_loc[1]

    distance = haversine(city[0], city[1], ISS_lat, ISS_long)

    return distance

dist = ISS_distance(Chicago)
print(f"The distance between Chicago and the ISS is currently {dist} miles.")


# ISS_distance tests

assert type(dist) == float

dist1 = ISS_distance(Chicago)
time.sleep(1)
dist2 = ISS_distance(Chicago)
Damascus = [33.5132, 36.2768]
dist3 = ISS_distance(Damascus)

print(dist1)
print(dist2)
print(dist3)


assert abs(dist1 - dist2) > 5 # check that the distance changes over time, given that ISS travels ~5 miles per second
assert abs(dist2 - dist3) > 3050 # check that two cities 6100 mi apart have distances at least 3050mi different at approximately the same time



#
# Open-ended possibility:  
#    (a) create a new function ISS_distance(place) that takes in a place name
#    (b) find a service by which you can look up the lat + CS35_Participant_2 using the place name
#         (b*)  I'm not sure how to do this - it's exploratory! 
#    (c) then, continue with the previous computation to find the ISS distance! :) 
#

# The final problem of this hw2 is to take on _ONE_ open-ended possibility. 
#     (This ISS-themed one is only the first possibility.)
#     Others, below, involve earthquakes, or your own choice of API exploration...


# Task 5 in blocks after 3,4





# hw2: USGS Tasks 3 and 4 ...
# 
# Two functions:  Quake_loop(), Quake_compare(place1, place2)

#
# Use the USGS (earthquake) examples above to write a function, named 
#     
#      Quake_loop()
#
# that uses requests within a loop of your own design in order to
#   + obtain at least 10 distinct, comparable data elements (counts are encouraged; other items ok)
#   + see the assignment page for an example where the looping iterates over the _month_
#
#   + choose your favorite parameter(s) to vary, e.g., magnitude, time, radius, location, etc.
#   + it should collect all of those data elements into a list
#   + and render the list in a neatly formatted chart (f-strings welcome; not required)
#
#   + in addition, include a overall reflection on the results, as well as a comment on additional effort
#     that could expand your results (you need not run it), and any concerns or caveats about the data...
#   + feel free to copy-paste-edit the markdown "reflection-template," above  


def Quake_loop():
    """loops thru radii from Claremont in increments of 100km and returns how many >= 2.3 magnitude earthquakes there were in the last year in a dictionary"""

    url = "https://earthquake.usgs.gov/fdsnws/event/1/count"
    min_mag = 2.3               # the minimum magnitude considered a quake (min_mag)
    start_time = "2025-01-01"   # this is the year-month-day format of the start
    finish_time = "2025-02-01"  # similar for the end
    radius_in_km = [100,200,300,400,500,600,700,800,900,1000]

    count_dict = {}

    for x in radius_in_km:
        param_dictionary = { "format":"geojson",
                            "starttime":start_time,
                            "endtime":finish_time,
                            "minmagnitude":min_mag,
                            "latitude":34.0967,
                            "longitude":-117.7198,
                            "maxradiuskm":x,
                            }
        result = requests.get(url, params=param_dictionary)
        quake_count = result.json()
        #print(quake_count)
        count_dict[x] = quake_count


    return count_dict

d = Quake_loop()
data = d

print('Result:')
print()
for x in d:
    print(f"At a radius {x} km from Claremont, there were {d[x]['count']} earthquakes of magnitude 2.3 or more in the first month of 2025.")




assert type(d[100]['count']) == int # should be integer count, no decimals

#check that greater radii result in a greater or equal count, this must always be true as a larger circle includes the previous one
prev_count = 0
for x in d:
    assert d[x]['count'] >= prev_count
    prev_count = d[x]['count']




# 
# Once your Quake_loop() function is working, write a new function
#
#       Quake_compare(place1, place2)
#
# where place1 should be a 2-element tuple:  (latitude1, longitude1)
# and  place2 should be a 2-element tuple:  (latitude2, longitude2)
#
# and then your function should compare which of the two places is "quakier" (not a real word)
# for a given time span (you choose), and a given strength-of-quakes (you choose), and
# for a specific radius around each of the two places (you choose)
#
# As is clear, there is lots of freedom to design a "comparison of quakiness" -- wonderful!
# Feel free to start somewhere, and tune the results.
#
# Your function should print the results it finds (in this case, it's not important to return
# and specific value -- though you're encouraged (not required) to create a helper function and 
# then call it twice for the two locations! (That helper function would need a return value!)
#

import json

def Quake_compare(place1, place2):
    """takes place tuples: (lat1, long1) and (lat2, long2) and returns the place which is quakier. 
    Quakier is defined as having a greater quake score, where score is the sum of the magnitude of any quake >= 3.0 magnitude that occured in the last five years within 250km of a place.
    The limit is set at 3.0 because that is the typical strength needed to feel an earthquake.
    """
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    min_mag = 3.0               # the minimum magnitude considered a quake (min_mag)
    start_time = "2020-02-12"   # this is the year-month-day format of the start
    finish_time = "2025-02-12"  # similar for the end
    radius_in_km = 250

    loc = [place1, place2]

    quakier = ''
    quake_scores = []  # Renamed from sum to quake_scores

    for x in loc:
        param_dictionary = { "format":"geojson",
                            "starttime":start_time,
                            "endtime":finish_time,
                            "minmagnitude":min_mag,
                            "latitude":x[0],
                            "longitude":x[1],
                            "maxradiuskm":radius_in_km,
                            }
        result = requests.get(url, params=param_dictionary)
        d = result.json()
        # nicer_string = json.dumps(d, indent=4)   # We can specify the indentation. 
        # print(nicer_string)

        mag_values = [feature['properties']['mag'] for feature in d['features']]
        # print(mag_values)
        # print(type(mag_values[1]))
        total = sum(mag_values)
        quake_scores.append(total)
        # print(total)

    if quake_scores[0] > quake_scores[1]:
        quakier = "Location 1, at " + str(place1) + " is quakier."
    elif quake_scores[0] == quake_scores[1]:
        quakier = 'The two locations are equally quaky.'
    else:
        quakier = "Location 2, at " + str(place2) + " is quakier."

    return quakier

manila = (14.5995, 120.9842)
mexicoCity = (19.4326, -99.1332)

comp = Quake_compare(manila, mexicoCity)





# Quake_compare tests

assert Quake_compare(manila, manila) == 'The two locations are equally quaky.'

Utsunomiya = (36.5551, 139.8826) # one of if not the most earthquake-prone city

assert Quake_compare(Utsunomiya, manila) == f'Location 1, at {Utsunomiya} is quakier.'
assert Quake_compare(manila, Utsunomiya) == f'Location 2, at {Utsunomiya} is quakier.'






# workspace for testing things out on their own

'''

import requests
from bs4 import BeautifulSoup

url = ""
result = requests.get(url)

if result.status_code == 200:
    # Parsing the HTML content
    soup = BeautifulSoup(result.content, 'html.parser')
    
    # Example: Extracting the article title
    title = soup.title.string
    print(f"Article Title: {title}")
    
    # Example: Extracting all paragraph texts
    paragraphs = soup.find_all('p')
    for para in paragraphs:
        print(para.text)
        print()
        print('XXXXXX')
else:
    print(f"Failed to retrieve content. Status code: {result.status_code}")

'''





#
# Cells for your own API experimentations + results!
#

def count_letters(word, paragraphs): #by copilot
    """takes in string of a word and paragraph, returns a sum of the number of times the word's letters appear in that paragraph"""
    # Initialize a dictionary to hold letter counts
    letter_counts = {letter: 0 for letter in word}
    
    # Loop through each paragraph
    for paragraph in paragraphs:
        for letter in word:
            letter_counts[letter] += paragraph.lower().count(letter.lower())
    
    return sum(letter_counts.values())

import requests
d = {}


def startUp_eval(url):
    """takes a url input for a random start up idea generator, 
    returns a % prediction of success based on zodiac associations of the elements of the idea
    """

    # first, obtain this for that idea from 'itsthisforthat.com' 
    result = requests.get(url)
    idea = result.json()
    d = idea
    this = d['this']
    that = d['that']

    # we then would like to evaluate the liklihood of financial and professional success of this business venture
    # ie, does the idea flow/harmonize/vibe
    # what better way to evaluate this than zodiac compatibilities (this is meant entirely for fun ofc not seriously)

    # need to assign 'this' and 'that' to their corresponding star sign
    # count the letter matches of this/that to the zodiac sign descriptions from glamour magazine
    # do the same for each star sign and find the closest count value match

    url = "https://www.glamourmagazine.co.uk/article/zodiac-sign-personality-traits"
    result = requests.get(url)
    full_text = ''

    if result.status_code == 200:
        # Parsing the HTML content
        soup = BeautifulSoup(result.content, 'html.parser')
        
        # Example: Extracting all paragraph texts
        paragraphs = soup.find_all('p')
        for para in paragraphs:
            # print(type(para.text))
            full_text = full_text + para.text

    else:
        print(f"Failed to retrieve content. Status code: {result.status_code}")

    this_score = count_letters(this, full_text)
    that_score = count_letters(that, full_text)

    aries = count_letters('aries', full_text)
    taurus = count_letters('taurus', full_text)
    gemini = count_letters('gemini', full_text)
    cancer = count_letters('cancer', full_text)
    leo = count_letters('leo', full_text)
    virgo = count_letters('virgo', full_text)
    libra = count_letters('libra', full_text)
    scorpio = count_letters('scorpio', full_text)
    sagittarius = count_letters('sagittarius', full_text)
    capricorn = count_letters('capricorn', full_text)
    aquarius = count_letters('aquarius', full_text)
    pisces = count_letters('pisces', full_text)

    zodiac_values = {'aries': aries, 
                     'taurus': taurus, 
                     'gemini': gemini, 
                     'cancer': cancer, 
                     'leo': leo, 
                     'virgo': virgo, 
                     'libra': libra, 
                     'scorpio': scorpio, 
                     'sagittarius': sagittarius, 
                     'capricorn': capricorn, 
                     'aquarius': aquarius, 
                     'pisces': pisces}

    closest_match_this = 1000000000
    closest_match_that = 1000000000

    this_zodiac = ''
    that_zodiac = ''

    for x in zodiac_values:
        offset_this = abs(this_score - zodiac_values[x])
        offset_that = abs(that_score - zodiac_values[x])
        if offset_this < closest_match_this:
            closest_match_this = offset_this
            this_zodiac = x
        if offset_that < closest_match_that:
            closest_match_that = offset_that
            that_zodiac = x

    # print(this_zodiac)
    # print(that_zodiac)

    # we now have a zodiac sign for each part of the start up idea, now need to determine compatability as a proxy for expected commercial success
    
    compatibility_scores = {
    'aries': {'aries': 70, 'taurus': 60, 'gemini': 80, 'cancer': 50, 'leo': 90, 'virgo': 40, 'libra': 75, 'scorpio': 55, 'sagittarius': 85, 'capricorn': 45, 'aquarius': 80, 'pisces': 50},
    'taurus': {'aries': 60, 'taurus': 70, 'gemini': 50, 'cancer': 85, 'leo': 60, 'virgo': 90, 'libra': 65, 'scorpio': 80, 'sagittarius': 50, 'capricorn': 95, 'aquarius': 55, 'pisces': 85},
    'gemini': {'aries': 80, 'taurus': 50, 'gemini': 70, 'cancer': 60, 'leo': 75, 'virgo': 65, 'libra': 90, 'scorpio': 50, 'sagittarius': 85, 'capricorn': 40, 'aquarius': 95, 'pisces': 60},
    'cancer': {'aries': 50, 'taurus': 85, 'gemini': 60, 'cancer': 70, 'leo': 65, 'virgo': 75, 'libra': 55, 'scorpio': 90, 'sagittarius': 50, 'capricorn': 80, 'aquarius': 45, 'pisces': 95},
    'leo': {'aries': 90, 'taurus': 60, 'gemini': 75, 'cancer': 65, 'leo': 70, 'virgo': 50, 'libra': 85, 'scorpio': 55, 'sagittarius': 95, 'capricorn': 45, 'aquarius': 80, 'pisces': 50},
    'virgo': {'aries': 40, 'taurus': 90, 'gemini': 65, 'cancer': 75, 'leo': 50, 'virgo': 70, 'libra': 60, 'scorpio': 85, 'sagittarius': 55, 'capricorn': 90, 'aquarius': 65, 'pisces': 80},
    'libra': {'aries': 75, 'taurus': 65, 'gemini': 90, 'cancer': 55, 'leo': 85, 'virgo': 60, 'libra': 70, 'scorpio': 50, 'sagittarius': 80, 'capricorn': 55, 'aquarius': 95, 'pisces': 65},
    'scorpio': {'aries': 55, 'taurus': 80, 'gemini': 50, 'cancer': 90, 'leo': 55, 'virgo': 85, 'libra': 50, 'scorpio': 70, 'sagittarius': 60, 'capricorn': 95, 'aquarius': 45, 'pisces': 90},
    'sagittarius': {'aries': 85, 'taurus': 50, 'gemini': 85, 'cancer': 50, 'leo': 95, 'virgo': 55, 'libra': 80, 'scorpio': 60, 'sagittarius': 70, 'capricorn': 45, 'aquarius': 90, 'pisces': 55},
    'capricorn': {'aries': 45, 'taurus': 95, 'gemini': 40, 'cancer': 80, 'leo': 45, 'virgo': 90, 'libra': 55, 'scorpio': 95, 'sagittarius': 45, 'capricorn': 70, 'aquarius': 60, 'pisces': 85},
    'aquarius': {'aries': 80, 'taurus': 55, 'gemini': 95, 'cancer': 45, 'leo': 80, 'virgo': 65, 'libra': 95, 'scorpio': 45, 'sagittarius': 90, 'capricorn': 60, 'aquarius': 70, 'pisces': 65},
    'pisces': {'aries': 50, 'taurus': 85, 'gemini': 60, 'cancer': 95, 'leo': 50, 'virgo': 80, 'libra': 65, 'scorpio': 90, 'sagittarius': 55, 'capricorn': 85, 'aquarius': 65, 'pisces': 70}
    } # courtesy of copilot

    compatibility_score = compatibility_scores[this_zodiac][that_zodiac]


    return this, that, compatibility_score



url = "https://itsthisforthat.com/api.php?json"
res = startUp_eval(url)

print(f"The idea was a {res[0]} for {res[1]}. This venture is predicted to have a {res[2]}% chance of success.")
print()
print("(disclaimer that some of the this/that combinations generated can be problematic, this is ofc not intended to be taken seriously whatsoever)")



