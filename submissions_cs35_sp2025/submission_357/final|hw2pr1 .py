#
# computing-styling trick of the day     (or, of the minute...)
#
# The setting for word-wrapping on the output is
#     "notebook.output.wordWrap": true,   (in your settings.json file or from Code ... Settings ...) 

print( list(range(100)) )



#
import requests
# see if we have the requests library...
#



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
CS35_Participant_2 = float(json_contents['iss_position']['longitude'])
json_contents['message']       # Challenge:  could we access the other components? What _types_ are they?!!


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
min_mag = 4.2               # the minimum magnitude considered a quake (min_mag)
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
min_mag = 6.5               # the minimum magnitude considered a quake (min_mag)
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
# That's hard to read!
# Let's pretty-print it with the json library
#       Also, this version can be pasted into online formatters, e.g., https://jsonformatter.org/

import json 
nice_string = json.dumps(d)   # this outputs a "nicely formatted string" using double quotes
print(nice_string)

 



import json 
nicer_string = json.dumps(d, indent=4)   # We can specify the indentation. 
print(nicer_string)                      # It's another tree structure... !


# API endpoint
url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Query parameters for the entire year of 2024
params = {
    "format": "geojson",  # Get data in JSON format
    "starttime": "2024-01-01",
    "endtime": "2024-12-31",
    "minmagnitude": 5.0,  # Get only significant earthquakes
    "limit": 1000,  # Get a large dataset
    "orderby": "magnitude"  # Order results by magnitude (largest first)
}

# Make API request
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()  # Convert response to JSON format
    earthquakes = data.get("features", [])

    if earthquakes:
        # Find the largest earthquake
        largest_quake = max(earthquakes, key=lambda eq: eq["properties"]["mag"])
        largest_info = {
            "Magnitude": largest_quake["properties"]["mag"],
            "Location": largest_quake["properties"]["place"],
            "Time (UTC)": largest_quake["properties"]["time"]
        }
        
        print("Largest Earthquake in 2024:")
        print(f"  - Magnitude: {largest_info['Magnitude']}")
        print(f"  - Location: {largest_info['Location']}")
        print(f"  - Time (Timestamp): {largest_info['Time (UTC)']}")

    else:
        print("No significant earthquakes found in 2024.")

else:
    print(f"Error: {response.status_code}")



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

from math import radians, sin, cos, sqrt, asin
import requests

def ISS_now():
    """ get the ISS current location and return lat CS35_Participant_2! """
    url = "http://api.open-notify.org/iss-now.json" 
    result = requests.get(url)  
    json_contents = result.json()
    CS35_Participant_2 = json_contents['iss_position']['longitude']
    lat = json_contents['iss_position']['latitude']
    return lat,CS35_Participant_2

lat, CS35_Participant_2 = ISS_now()
print(f"{lat = } and {CS35_Participant_2 = }")




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
# lat and CS35_Participant_2 of Claremont and ISS
def haversine(lat1, long1, lat2, long2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lat1, long1, lat2, long2 = map(float, [lat1, long1, lat2, long2])
    long1, lat1, long2, lat2 = map(radians, [long1, lat1, long2, lat2])

    # haversine formula
    dlong = long2 - long1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlong/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth. Use 3956 for miles. 6371 for km.
    r = 3956
    return c * r

def ISS_distance():
    lat_C = 34.097
    long_C = -117.719
    lat_ISS , long_ISS = ISS_now()
    result = haversine(lat_C, long_C, lat_ISS, long_ISS)
    return result 

print(f"Distance between Claremont and ISS: {ISS_distance()}")

#print(ISS_distance())



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


#
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

import requests
import time


def get_num_quakes(month, magnitude):
   """ returns the number of quakes in month (of '24)
           of at least magnitude ...
   """


   url = "https://earthquake.usgs.gov/fdsnws/event/1/count"


   # Let's use variables for three of the parameters:
   min_mag = magnitude         # the minimum magnitude considered a quake (min_mag)
   start_time = f"2024-{month:02d}-01"   # this is the year-month-day format of the start
   finish_time = f"2024-{month+1:02d}-01"  # similar for the end (f-strings! :)


   # we assemble a dictionary of our parameters, let's name it param_dictionary
   # there are many more parameters available. The problems below ask you to explore them...
   param_dictionary = { "format":"geojson",         # this is simply hard-coded to obtain json
                        "starttime":start_time,
                        "endtime":finish_time,
                        "minmagnitude":min_mag,
                       }


   # Here, we use requests to make the request. The parameters will be added by this API call:
   time.sleep(2)
   result = requests.get(url, params=param_dictionary)
   # print(f"result is {result}")                     # hopefully, this is 200
   # print(f"the full url used: {result.url}")   # this will include the parameters!
   d = result.json()
   number_of_quakes = d['count']
   month_names = {1: 'Jan', 2: 'Feb', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'Aug', 9:'Sep', 10: 'Oct'}
   print(f"{month_names[month]:<12} {number_of_quakes:<12}")
   return number_of_quakes

def Quake_loop():
   print(f"{'Month':<12} {'Quakes':<12}")
   print("-"*30)
   LoQ = []
   for month in range(1,11):
      magnitude = 6
      result = get_num_quakes(month,magnitude)
      LoQ += [result]
   return LoQ

print(Quake_loop())
    


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
#
import requests
import time
place1 = (34.0522, -118.2437)  # Los Angeles, CA
place2 = (35.6895, 139.6917)   # Tokyo, Japan

def get_quake_count(latitude, longitude, radius, min_magnitude, start_time, end_time):
    """ Returns the number of earthquakes near a given location within a given radius, time frame, and magnitude. """
    url = "https://earthquake.usgs.gov/fdsnws/event/1/count"

    params = {
        "format": "geojson",
        "starttime": start_time,
        "endtime": end_time,
        "minmagnitude": min_magnitude,
        "latitude": latitude,
        "longitude": longitude,
        "maxradiuskm": radius,
    }

    time.sleep(2)  # To avoid hitting API limits
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return data.get("count", 0)
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

def Quake_compare(place1, place2):
    """ Compares earthquake activity between two locations. """

    # Parameters
    min_magnitude = 5.0  # Consider quakes with magnitude >= 5.0
    radius = 500  # Search within 500 km of each location
    start_time = "2024-01-01"  # Start of 2024
    end_time = "2024-12-31"  # End of 2024

    # Get earthquake counts
    count1 = get_quake_count(place1[0], place1[1], radius, min_magnitude, start_time, end_time)
    count2 = get_quake_count(place2[0], place2[1], radius, min_magnitude, start_time, end_time)

    # Print results
    print(f"Earthquake count near location 1 (Lat: {place1[0]}, Lon: {place1[1]}): {count1}")
    print(f"Earthquake count near location 2 (Lat: {place2[0]}, Lon: {place2[1]}): {count2}")

    if count1 > count2:
        print("Location 1 is quakier!")
    elif count2 > count1:
        print("Location 2 is quakier!")
    else:
        print("Both locations have the same earthquake activity.")

Quake_compare(place1, place2)





import http.client

conn = http.client.HTTPSConnection("exercisedb.p.rapidapi.com")

headers = {
    'x-rapidapi-key': "ea05327a5dmsha02021e1aacef7ap1919c5jsne180af786525",
    'x-rapidapi-host': "exercisedb.p.rapidapi.com"
}

conn.request("GET", "/status", headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))


#
# Cells for your own API experimentations + results!
# using color API and excercise DB to personalize workout recommendation based on the user input color.  

import requests
import http.client
import json

# Mood-Color to Workout Mapping (Grouped by Similar Colors)
MOOD_WORKOUT_MAP = {
    "Red": ["cardio", "HIIT", "boxing"],        # High intensity, aggressive
    "Blue": ["stretching", "yoga", "pilates"],  # Calm, relaxed
    "Yellow": ["plyometrics", "dance"],        # Energetic, fun
    "Green": ["running", "hiking"],            # Nature, endurance
    "Purple": ["strength", "powerlifting"],    # Focus, strength training
    "Orange": ["HIIT", "cardio"],              # Explosive energy
    "Pink": ["pilates", "yoga"],               # Balanced, mindful
    "Black": ["boxing", "combat"],             # Power, combat sports
    "White": ["meditation", "stretching"],     # Low intensity, mindfulness
}
def get_color_details(hex_code):
    url = f"https://www.thecolorapi.com/id?hex={hex_code.strip('#')}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return {
            "name": data['name']['value'],  
            "rgb": data['rgb']  # Gets RGB values
        }
    
    return {"name": "Unknown", "rgb": {"r": 255, "g": 0, "b": 0}}  # Default to red

# Function to get color name from The Color API
def get_color_name(hex_code):
    url = f"https://www.thecolorapi.com/id?hex={hex_code.strip('#')}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()['name']['value']
    
    return "Unknown"

# maps RGB color to closest base color. Since color api has specific color names, 
#we use the hex color to classify the colors

def get_closest_mood_color(rgb):
    r, g, b = rgb["r"], rgb["g"], rgb["b"]

    if r > 200 and g < 100 and b < 100:
        return "Red"
    elif b > 200 and r < 100 and g < 150:
        return "Blue"
    elif r > 200 and g > 200 and b < 100:
        return "Yellow"
    elif g > 150 and r < 150 and b < 100:
        return "Green"
    elif r > 150 and b > 150 and g < 100:
        return "Purple"
    elif r > 200 and g > 120 and b < 50:
        return "Orange"
    elif r > 200 and g < 150 and b > 150:
        return "Pink"
    elif r < 50 and g < 50 and b < 50:
        return "Black"
    elif r > 200 and g > 200 and b > 200:
        return "White"
    
    return "Red"  # Default fallback

# Function to get workouts from ExerciseDB API
def get_workouts(workout_type):
    url = f"https://exercisedb.p.rapidapi.com/exercises/bodyPart/{workout_type}"
    headers = {
        'x-rapidapi-key': "ea05327a5dmsha02021e1aacef7ap1919c5jsne180af786525",
        'x-rapidapi-host': "exercisedb.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        exercises = response.json()
        if isinstance(exercises, list) and len(exercises) > 0:
            return [{"name": ex["name"], "gif": ex["gifUrl"]} for ex in exercises[:5]]
    
    return [{"name": "Jumping Jacks", "gif": "https://example.com/jumping-jacks.gif"}]  # Default fallback

# Function to generate workout based on color mood
def mood_based_workout(hex_code):
    print("\n🎨 Checking color mood...")
    color_details = get_color_details(hex_code)
    detected_color_name = color_details["name"]
    detected_rgb = color_details["rgb"]

    print(f"✅ Detected Color: {detected_color_name}")

    # Find the closest mood color based on RGB
    mood_color = get_closest_mood_color(detected_rgb)
    print(f" Matched Mood Color: {mood_color}")

    # Pick a workout type from the available ones for that mood
    workout_types = MOOD_WORKOUT_MAP[mood_color]
    selected_workout = workout_types[0]  # Always pick the first workout type
    
    print(f"Suggested Workout Type: {selected_workout.capitalize()}")

    # Fetch exercises from API
    print("\n⏳ Fetching personalized exercises...\n")
    exercises = get_workouts(selected_workout)

    # Display results
    print(" Recommended Exercises")
    for i, ex in enumerate(exercises, 1):
        print(f"{i}. {ex['name']}")
        print(f"   🏋️‍♂️ Exercise GIF: {ex['gif']}\n")

#  User Input for Hex Code
hex_code = input("🎨 Enter a color hex code (e.g., #FF5733): ")
mood_based_workout(hex_code)


#Output format was with the help of ChatGPT




