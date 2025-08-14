#
# computing-styling trick of the day     (or, of the minute...)
#
# The setting for word-wrapping on the output is
#     "notebook.output.wordWrap": true,   (in your settings.json file or from Code ... Settings ...) 

print( list(range(10)) )



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
result.url


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

json_contents['message']       # Challenge:  could we access the other components? What _types_ are they?!!


json_contents['timestamp']


CS35_Participant_2 = float(json_contents['iss_position']['longitude'])
CS35_Participant_2


#
# In Python, we can use the resulting dictionary... let's see its keys:
print(list(json_contents.keys()))  

# Also, watch out for string vs. numeric types, e.g., for latitude and longitude.
# At heart, _all_ web data are strings... .

# These experiments will be helpful for problem 1, below :)


lat = float(json_contents['iss_position']['latitude'])



def haversine(lat1, long1, lat2, long2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    from math import sin, cos, radians, asin, sqrt   # this import is for the sin, cos, radians, ...
    # convert decimal degrees to radians

    long1, lat1, long2, lat2 = map(radians, [long1, lat1, long2, lat2])

    # haversine formula
    dlong = long2 - long1
    dlat = lat2 - lat1
    trig = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlong/2)**2
    # Radius of earth. Use 3956 for miles. 6371 for km.
    radius = 3956  # we'll use miles!
    return radius * 2 * asin(sqrt(trig))

haversine(-42.6203,-8.7386,34.0967,-117.719)


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
list(dictionary.keys()) # How do we get 'Aliiien' from newdata?


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
print(text[:242])                  # we'll print the first few characters...  

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
#     + in Dec 2025
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
min_mag = 6.5 
max_mag = 7.0              # the minimum magnitude considered a quake (min_mag)
start_time = "2025-01-01"   # this is the year-month-day format of the start
finish_time = "2025-02-01"  # similar for the end

# we assemble a dictionary of our parameters, let's name it param_dictionary
# there are many more parameters available. The problems below ask you to explore them...
param_dictionary = { "format":"geojson",         # this is simply hard-coded to obtain json
                     "starttime":start_time,
                     "endtime":finish_time,
                     "minmagnitude":min_mag,
                     "maxmagnitude":max_mag
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
print(nicer_string)                      # It's another tree structure... !


#
# hw2: ISS tasks 1 and 2 ...
# 
# Two functions:  ISS_now(), ISS_distance()
import requests

url = "http://api.open-notify.org/iss-now.json"   # this is sometimes called an "endpoint" ...
result = requests.get(url)
json_contents = result.json()

#
# Use the ISS examples above to write a function, named 
#     
#      ISS_now()
#
# that uses requests to return the current latitude and longitude -- as floating-point values -- right now.
# Be sure to test it! 

def ISS_now():
    lat1 = float(json_contents['iss_position']['latitude'])
    long1 = float(json_contents['iss_position']['longitude'])
    return lat1, long1

ISS_now()



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
def ISS_distance():
    """
    Calculate the distance of the ISS at any given time to Manila, Philippines
    """
    from math import sin, cos, radians, asin, sqrt   # this import is for the sin, cos, radians, ...
    # convert decimal degrees to radians

    lat1, long1 = ISS_now()
    # Hardcoded the latitude and longitude of Manila, Philippines
    lat2 = 14.5995
    long2 = 120.9842

    long1, lat1, long2, lat2 = map(radians, [long1, lat1, long2, lat2])

    # haversine formula
    dlong = long2 - long1
    dlat = lat2 - lat1
    trig = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlong/2)**2
    
    # Radius of earth. Use 3956 for miles. 6371 for km.
    radius = 3956  # we'll use miles!
    return radius * 2 * asin(sqrt(trig))

distance = ISS_distance()

# Print the result
print(f"Distance of ISS from Manila, PH: {distance:.2f} miles")



#
# Open-ended possibility:  
#    (a) create a new function ISS_distance(place) that takes in a place name
#    (b) find a service by which you can look up the lat + CS35_Participant_2 using the place name
#         (b*)  I'm not sure how to do this - it's exploratory! 
#    (c) then, continue with the previous computation to find the ISS distance! :) 
#
d = {}

import requests
from math import sin, cos, radians, asin, sqrt

def ISS_distance(place):

    url = f"https://api.geodatasource.com/format=json&"
    result = requests.get(url).json()
    
    if result:
        lat = d[place]
        CS35_Participant_2 = float(result[0]['CS35_Participant_2'])
        return lat, CS35_Participant_2
    
place_name = "Manila"
distance = ISS_distance(place_name)

print(f"Distance of ISS from {place_name}: {distance:.2f} miles")

# The final problem of this hw2 is to take on _ONE_ open-ended possibility. 
#     (This ISS-themed one is only the first possibility.)
#     Others, below, involve earthquakes, or your own choice of API exploration...


import requests
import time

def get_num_quakes(start_time, finish_time, min_mag, latitude, longitude, max_radius_km):
   """ returns the number of quakes in month (of '24)
           of at least magnitude ...
   """
   url = "https://earthquake.usgs.gov/fdsnws/event/1/count"
 
   # we assemble a dictionary of our parameters, let's name it param_dictionary
   # there are many more parameters available. The problems below ask you to explore them...
   param_dictionary = { "format":"geojson",         # this is simply hard-coded to obtain json
                        "starttime":start_time,
                        "endtime":finish_time,
                        "minmagnitude":min_mag,
                        "latitude": latitude,         
                        "longitude": longitude,      
                        "maxradiuskm": max_radius_km
                       }
   
   result = requests.get(url, params=param_dictionary)
   # return f"({result} the result.url is {result.url})"
   d = result.json()
   count = d.get('count')
   return count

# Manila to rest of Luzon is a 200 km radius but max_radius_km max is 180
print(get_num_quakes("2024-01-01","2024-04-01",4.0,14.5995,120.9842,180))
print(get_num_quakes("2024-01-01","2024-04-01",4.0,34.0967,-117.7198,180))


#
# hw2: USGS Tasks 3 and 4 ...
# 
# Two functions:  Quake_loop(), Quake_compare(place1, place2)

#
# Use the USGS (earthquake) examples above to write a function, named 
#     
import requests
import time

magnitudes = [2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0] # the strongest earthquake magnitude was 9.5 in 1960!
LoQ = []

def Quake_loop():
    ''' loops over different magnitudes to collect earthquake counts ''' 
    start_time = "2024-01-01"   
    finish_time = "2024-04-01"
    
    latitude = 14.5995 #34.0967 
    longitude = 120.9842 #-117.7198 
    max_radius_km = 180  

    LoQ = []

    for magnitude in magnitudes:
        count = get_num_quakes(start_time, finish_time, magnitude, latitude, longitude, max_radius_km)
        LoQ.append(count)
        
    return LoQ
LoQ = Quake_loop()

print(f"Between 2024-01-01 and 2024-04-01, there were the following earthquakes")
for i, magnitude in enumerate([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]):
    print(f"Magnitude {magnitude}: {LoQ[i]} quakes")
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


# 
# Once your Quake_loop() function is working, write a new function
#
def Quake_compare(place1, place2):
    ''' determines which of two places is quakier based on specified time span, magnitude, and radius around the two chosen places'''
#
# and then your function should compare which of the two places is "quakier" (not a real word)
# for a given time span (you choose), and a given strength-of-quakes (you choose), and
# for a specific radius around each of the two places (you choose)
#
    start_time = f"2024-01-01"   
    finish_time = f"2024-04-01"
    min_mag = 4.0
    max_radius_km = 180  

    lat1, long1 = place1 
    lat2, long2 = place2

    count1 = get_num_quakes(start_time, finish_time, min_mag, lat1, long1, max_radius_km)
    count2 = get_num_quakes(start_time, finish_time, min_mag, lat2, long2, max_radius_km)

    print(f"Between {start_time} and {finish_time}, with magnitude ≥ {min_mag} and radius ≤ {max_radius_km} km:")
    print(f"Claremont {place1} had {count1} quakes.")
    print(f"Manila {place2} had {count2} quakes.")

    if count1 > count2:
        print(f"Claremont {place1} is quakier than Manila {place2}.")
    elif count2 > count1:
        print(f"Manila {place2} is quakier than Claremont {place1}.")
    else:
        print(f"Claremont {place1} and Manila {place2} are just as quaky.")

place1 = (34.0967,-117.7198) # Claremont
place2 = (14.5995,120.9842) # Manila

Quake_compare(place1,place2)

# As is clear, there is lots of freedom to design a "comparison of quakiness" -- wonderful!
# Feel free to start somewhere, and tune the results.
#
# Your function should print the results it finds (in this case, it's not important to return
# and specific value -- though you're encouraged (not required) to create a helper function and 
# then call it twice for the two locations! (That helper function would need a return value!)
#
#


#
import requests

# 

def get_ye_quotes():
   """ returns a funfact about the number of letters in a random kanye west quote ...
   """
   url = "https://api.kanye.rest"

   result = requests.get(url)
   # return f"({result} the result.url is {result.url})"
   d = result.json()
   quote = d.get('quote')
   print(quote)
   number = str(len(quote)) 
   
   url2 = f"http://numbersapi.com/{number}/trivia"

   result2 = requests.get(url2)
   print(result2.text) # get plain text

get_ye_quotes()

#


