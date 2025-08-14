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
def getLong(contents):
    CS35_Participant_2 = float(contents['iss_position']['longitude'])       # Challenge:  could we access the other components? What _types_ are they?!!
    return CS35_Participant_2

def getLat(contents):
    lat = float(contents['iss_position']['latitude'])       # Challenge:  could we access the other components? What _types_ are they?!!
    return lat

print(f"The ISS Longitude is {getLong(json_contents)}.")
print(f"The ISS Latitude is {getLat(json_contents)}.")


#
# In Python, we can use the resulting dictionary... let's see its keys:
print(list(json_contents.keys()))  

# Also, watch out for string vs. numeric types, e.g., for latitude and longitude.
# At heart, _all_ web data are strings... .

# These experiments will be helpful for problem 1, below :)


from math import *    # this import is for the sin, cos, radians, ...

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


claremontLong = -117.7198
claremontLat = 34.0967

issLong = getLong(json_contents)
issLat = getLat(json_contents)

print(f"The haversine distance between claremont and the ISS is {(haversine(claremontLat, claremontLong, issLat, issLong))/1.609} miles.")


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

dictionary['key']


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




d['people'][4]['name'][3:0:-2]


d['people'][0]['name'][7:9]


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
min_mag = 4.2               # the minimum magnitude considered a quake (min_mag)
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
url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

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
print(f"There were {quake_count['metadata']['count'] } total quakes of magnitude greater than 2.2 within 300km of Claremont since the start of the year.")


#
# Here is the endpoint
#
url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Let's use variables for three of the parameters:
min_mag = 7.4               # the minimum magnitude considered a quake (min_mag)
start_time = "2024-01-01"   # this is the year-month-day format of the start
finish_time = "2024-12-31"  # similar for the end

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
print(nicer_string)                      # It's another tree structure... !


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

def ISS_now():

    url = "http://api.open-notify.org/iss-now.json"   # this is sometimes called an "endpoint" ...
    result = requests.get(url)
    json_contents = result.json()
    #print(json_contents)
    lat = float(getLat(json_contents))
    CS35_Participant_2 = float(getLong(json_contents))
    #print(json_contents)
    return (lat, CS35_Participant_2)

print(ISS_now())



# 
# Once your ISS_now() function is working, write a new function
#

def ISS_distance(): 
#
# which uses ISS_now to obtain the lat/CS35_Participant_2 of the ISS and then
# uses the haversine distance (look up a Python implementation or use one of ours... :)
# to compute the ISS's distance from a city of your choice.
#
# The haversine distance computes the "great circle" distance from two points on the globe
#     using latitude and longitude  
#
    ISS_loc = ISS_now()
    ISSLong = ISS_loc[1]
    ISSLat = ISS_loc[0]
    ClaremontLat = 34.0967
    ClaremontLong = -117.7198
    dist = haversine(ClaremontLat, ClaremontLong, ISSLat, ISSLong)
    return dist

print(ISS_distance())


#
# Open-ended possibility:  
#    (a) create a new function ISS_distance(place) that takes in a place name
#    (b) find a service by which you can look up the lat + CS35_Participant_2 using the place name
#         (b*)  I'm not sure how to do this - it's exploratory! 
#    (c) then, continue with the previous computation to find the ISS distance! :) 
#

# 
# Once your ISS_now() function is working, write a new function
#

def ISS_distanceFrom(place): 
#
# which uses ISS_now to obtain the lat/CS35_Participant_2 of the ISS and then
# uses the haversine distance (look up a Python implementation or use one of ours... :)
# to compute the ISS's distance from a city of your choice.
#
# The haversine distance computes the "great circle" distance from two points on the globe
#     using latitude and longitude  
#
    ISS_loc = ISS_now()
    ISSLong = ISS_loc[1]
    ISSLat = ISS_loc[0]
    placeLat = coordinatesfromName(place)[0]
    #print(placeLat)
    placeLong = coordinatesfromName(place)[1]
    #print(placeLong)
    
    dist = haversine(placeLat, placeLong, ISSLat, ISSLong)
    return dist

city = 'Claremont, CA'
d_from_claremont = ISS_distanceFrom(city)
print(f"The ISS is {d_from_claremont} miles from {city}.")

# The final problem of this hw2 is to take on _ONE_ open-ended possibility. 
#     (This ISS-themed one is only the first possibility.)
#     Others, below, involve earthquakes, or your own choice of API exploration...


from opencage.geocoder import OpenCageGeocode

def coordinatesfromName(placeName):
    #
    # uses the OpenCageGeocode API to return the latitude and longitude of place as a tuple given a place name 
    #
    OCG = OpenCageGeocode('77532d6bcfed4afdb6d45f163c26724f')
    results = OCG.geocode(placeName, language='en')
    #lat = results[0]['annotations']['lat']
    #CS35_Participant_2 = results[0]['annotations']['CS35_Participant_2']
    return (results[0]['bounds']['northeast']['lat'], results[0]['bounds']['northeast']['lng'])

coordinatesfromName('Almaty, Kazakhstan')




#
# hw2: USGS Tasks 3 and 4 ...
# 
# Two functions:  Quake_loop(), Quake_compare(place1, place2)

#
# Use the USGS (earthquake) examples above to write a function, named 
#

import matplotlib.pyplot as plt

url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
# Let's use variables for three of the parameters:
min_mag = 2.2               # the minimum magnitude considered a quake (min_mag)
start_time = "2024-01-01"   # similar, but for a year-CS35_Participant_2 span...
finish_time = "2025-01-01"  # similar for the end
radius_in_km = 300

def Quake_loop():  
    #
    #   Function that uses requests within a loop of your own design in order to
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

    lat = 37.668819
    CS35_Participant_2 = -122.080795
    count_list = []
    long_list = []
    for i in range(46):
        # we assemble a dictionary of our parameters, let's name it param_dictionary
        # there are many more parameters available. The problems below ask you to explore them...
        param_dictionary = { "format":"geojson",         # this is simply hard-coded to obtain json
                            "starttime":start_time,
                            "endtime":finish_time,
                            "minmagnitude":min_mag,
                            "latitude":lat,
                            "longitude":CS35_Participant_2,
                            "maxradiuskm":radius_in_km,
                            }
        result = requests.get(url, params = param_dictionary) 
        quake_info = result.json()
        count = quake_info['metadata']['count']
        long_list += [CS35_Participant_2]
        count_list += [count]
        CS35_Participant_2 += 1
    return (count_list, long_list)

quakes = Quake_loop() 
categories = quakes[1]
values = quakes[0]
plt.bar(categories, values, color = 'lightblue')
plt.xlabel("Longitude")
plt.ylabel("Number of Earthquakes")
plt.title("Longitude vs Number of Earthquakes Across the Continental US")
plt.show()
    



# 
# Once your Quake_loop() function is working, write a new function
#
def Quake_compare(place1, place2):
    places = [place1, place2]
    radius_in_km = 200
    min_mag = 4.0 
    max_count = 0 
    for e in places: 
        param_dictionary = { "format":"geojson",         # this is simply hard-coded to obtain json
                                "starttime":start_time,
                                "endtime":finish_time,
                                "minmagnitude":min_mag,
                                "latitude":e[0],
                                "longitude":e[1],
                                "maxradiuskm":radius_in_km,
                                }
        result = requests.get(url, params = param_dictionary) 
        quake_info = result.json()
        count = quake_info['metadata']['count']
        if count > max_count:
            max_count = count
            maxPlace = e 
    return maxPlace 

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
#



#Testing Quake_compare -- as expected when looking at the above graph, at latitude 37, there are many more earthquakes towards the Western US (-120 longitude) as opposed to the east coast (-80 longitude)
place1 = (37.668819, -120)
place2 = (37.668819, -80)
a = Quake_compare(place1, place2)
print(f"Between the two locations {place1} and {place2}, {Quake_compare(place1, place2)} is 'quakier'...")


#
# Cells for your own API experimentations + results!
#
# The function finds the number of minutes between the end of civil and nautical twilight in a given location and offers an interesting fact about that integer. 
#

def factAboutTwilightMins(city):
    url = 'https://api.sunrise-sunset.org/json'
    lat, lng = coordinatesfromName(city)
    param_dictionary = {'lat': lat, 'lng': lng}
    result = requests.get(url, params = param_dictionary)
    result = result.json()
    civilEnd = result['results']['civil_twilight_end'][0:5]
    if civilEnd[1] == ':':
        civilHour = float(civilEnd[0])
        civilMins = float(civilEnd[2:4])
    else:
        civilHour = float(civilEnd[0:2])
        civilMins = float(civilEnd[3:5])
    nauticalEnd = result['results']['nautical_twilight_end'][0:5]
    if nauticalEnd[1] == ':':
        nauticalHour = float(nauticalEnd[0])
        nauticalMins = float(nauticalEnd[2:4])
    else:
        nauticalHour = float(nauticalEnd[0:2])
        nauticalMins = float(nauticalEnd[3:5])
    #print(civilHour, civilMins, nauticalHour, nauticalMins)
    #print(result)
    numMins = int(60*(nauticalHour+(1/60)*nauticalMins- civilHour - (1/60)*civilMins))
    print(f'There are {numMins} minutes between the end of civil and nautical twilight in {city} today.') 
    url = f'http://numbersapi.com/{numMins}/trivia'
    result = requests.get(url)
    print(f'Interestingly: {result.text}')

factAboutTwilightMins('Claremont, CA')
factAboutTwilightMins('Belmont, MA')



