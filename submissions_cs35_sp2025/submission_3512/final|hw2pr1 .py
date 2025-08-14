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

json_contents['iss_position']['longitude']       # Challenge:  could we access the other components? What _types_ are they?!!


#
# In Python, we can use the resulting dictionary... let's see its keys:
print(list(json_contents.keys()))  

# Also, watch out for string vs. numeric types, e.g., for latitude and longitude.
# At heart, _all_ web data are strings... .

# These experiments will be helpful for problem 1, below :)


import requests
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

url = "http://api.open-notify.org/iss-now.json"  
# result = requests.get(url)
# result    
# <Response [200]>!

#ISS latitude & longitude
ISSlat = float(json_contents['iss_position']['latitude'])
ISSlong = float(json_contents['iss_position']['longitude'])

#Claremont latitude & longitude
Clarelat = 34.0967
Clarelong = -117.7198

print(f"The ISS is {haversine(ISSlat, ISSlong, Clarelat, Clarelong)} miles away from Claremont!")



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

dictionary['key']     # value
dictionary.get('fave')
dictionary['list'][3]['mascot']



import requests 

# here, we will obtain plain-text results from a request
#url = "https://www.cs.hmc.edu/~dodds/demo.html"  # try it + source
# url = "https://www.scrippscollege.edu/"          # another possible site...
# url = "https://www.pitzer.edu/"                  # another possible site...
url = "https://www.cmc.edu/"                     # and another!
# url = "https://www.cgu.edu/"
result = requests.get(url)        
print(f"result is {result}")        # hopefully it's 200


# if the request was successful, the Response will be [200]. 
# Then, we can grab the text - or json - from the site:

text = result.text                  # provides the HTML page as a large string...
print(f"len(text) is {len(text)}")  # let's see how large the HTML page is... 

#print("\nThe first 242 characters are\n")
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
min_mag = 6.5               # the minimum magnitude considered a quake (min_mag)
start_time = "2025-01-01"   # this is the year-month-day format of the start
finish_time = "2025-02-10"  # similar for the end

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


# ISS now
import requests

def ISS_now(url): 
    resultnow = requests.get(url) 
    ISScoord = resultnow.json()  
    # checked that data was successfully retrieved.. <Response [200]>!

    #ISS latitude & longitude
    ISSlat = float(ISScoord['iss_position']['latitude'])         # float allows function to be used below
    ISSlong = float(ISScoord['iss_position']['longitude'])
    
    return ISSlat, ISSlong

url = "http://api.open-notify.org/iss-now.json" #ISS now endpoint

ISSlat, ISSlong = ISS_now(url)

print(f"The current latitude of the ISS is {ISSlat} and the current longitude is {ISSlong}")



# ISS distance
from haversine import haversine
import requests

def ISS_distance(city):
    url = "http://api.open-notify.org/iss-now.json"  # same endpoint as above
    ISSlat, ISSlong = ISS_now(url)                   # extract latitude and longitude
    ISS = [ISSlat, ISSlong]

    # I tried to allow for users to input a city, but I started running into problems with API keys and websites blocking automated API calls :(
    # cityurl = "https://nominatim.openstreetmap.org/search?q="+placeinput+"&format=json"
    # city_req = requests.get(cityurl)
    # city_result = city_req.json() 

    # citylat = city_result['lat']
    # citylong = city_result['lon']

    return haversine(ISS, city)


# placeinput = input("To which city do you want to find the distance to the ISS? ")
# since the above ^^^ didn't work...
# instead I went with London 
londonlat = 51.509865
londonlong = -0.118092
london = [londonlat, londonlong]

distance = ISS_distance(london)

print(f"The distance from London to the ISS is {distance} kilometers")


# Quake_loop()
import requests
import time

def Quake_loop(year, lat, lon):
   """ returns largest quake in a year in any area"""
   
   url = "https://earthquake.usgs.gov/fdsnws/event/1/query"   

   min_mag = 2                      # the minimum magnitude considered (min_mag)
   start_time = f"{year}-01-01"     
   finish_time = f"{year+1}-01-01"  


   # dictionary of paramenters
   param_dictionary = { "format":"geojson",         
                        "starttime":start_time,
                        "endtime":finish_time,
                        "minmagnitude":min_mag,
                        "latitude": lat,
                        "longitude": lon,
                        "maxradiuskm": 300,
                       }


   maglist = []

   time.sleep(2)                                          # allows time between requests to not burden server 
   result = requests.get(url, params=param_dictionary)    # modifies endpoint according to parameters
   d = result.json()

   # I originally struggled with accessing the json data i wanted, because I was treating the dictionary like a list
   # here was my original code

   # for i in range(0, int(d['metadata']['count'])):
   #       name_of_quake = str(d[i]['features']['properties']['place'])
   #       magnitude = float(d[i]['features']['properties']['mag'])

   # it kept showing keyerror: 0, and I couldn't figure out why
   # in hindsight a dictionary isn't ordered like a list, and there is no "0" key....

   # chat helped me with the first three lines of the loop, and extracted the list within the dictionary for me to work with
   for q in d['features']:
      name_of_quake = q['properties']['place']           # extracts location of quake
      magnitude = q['properties']['mag']                 # extracts magnitude of quake
      maglist.append(magnitude)
      if magnitude == max(maglist):                      # finds largest magnitude quake that year, stores magnitude & place
         maxmag = magnitude
         maxname = name_of_quake
   
   return maxmag, maxname

print(f"What were the largest earthquakes in a 300km radius of Portland, OR every year since 2015?")
print(" ")

lat = 45.523064
lon = -122.676483

for year in range(2015,2025):
   maxmag, maxname = Quake_loop(year, lat, lon)
   print(f"The largest quake in {year} was {maxname}, with a magnitude of {maxmag}")
   print(" ")


# compares the largest quakes from the last ten years in two locations
import requests
import time

def Quake_compare(LA, PDX):
   countLA = 0
   countPDX = 0
   
   for year in range(2015,2025):                                 # each year, over last ten years
      maxmagLA, maxnameLA = Quake_loop(year, LA[0], LA[1])       # find largest LA quake
      maxmagPDX, maxnamePDX = Quake_loop(year, PDX[0], PDX[1])   # and largest Portland quake

      if maxmagLA >= maxmagPDX:       # if largest quake was in LA, keep count of that
         countLA +=1
      else: 
         countPDX +=1                 # also keep count if largest quake was in Portland
   return countLA, countPDX

LA = [34.052235,-118.243683]
PDX = [45.523064, -122.676483]

countLA, countPDX = Quake_compare(LA, PDX)

if countLA > countPDX:      # now compare which had the larger quake over all ten years
   print(f"There were more large earthquakes in LA than Portland")
else:  print(f"There were more large earthquakes in Portland than LA")


# preevolution chain 
import requests

def Evolves_from(pokemon):
    url= f'https://pokeapi.co/api/v2/pokemon-species/{pokemon}'       # modifies endpoint according to pokemon of choice
    result = requests.get(url)
    pokemond = result.json()
    #print(pokemon)

    if pokemond["evolves_from_species"] == None:                      # sees if your pokemon has a preevolution or not
        prepokemon = None
    else: prepokemon = pokemond["evolves_from_species"]['name']

    return prepokemon

pokemon = input("What pokemon do you want to find the preevolution(s) of? ")

while Evolves_from(pokemon) != None:                             # loops the function until there are no more preevolutions & you reached base evolution
    print(f"{pokemon} evolves from {Evolves_from(pokemon)}")
    pokemon = Evolves_from(pokemon)

if Evolves_from(pokemon) == None:
    print(f"{pokemon} does not have a pre-evolution")


# getting "flavor text" (small blurbs about a pokemon) for pokemon & its preevolution chain

def flavor_text(pokemon):
    url= f'https://pokeapi.co/api/v2/pokemon-species/{pokemon}'        # modifies endpoint according to pokemon of choice
    result = requests.get(url)
    pokemond = result.json()
    
    flavortext = pokemond["flavor_text_entries"][0]['flavor_text']     # extracts flavor text

    return flavortext

pokemon = input("What pokemon, and its preevolution(s), do you want to know more about? ")

flavortext = flavor_text(pokemon)       # prints flavor text of originally selected pokemon
print(f"{pokemon}: {flavortext}")
print(" ")

while Evolves_from(pokemon) != None:                    # loops while your pokemon has preevolutions          
    flavortext = flavor_text(Evolves_from(pokemon))     # to find their flavor texts as well

    print(f"{Evolves_from(pokemon)}: {flavortext}")
    print(" ")
    pokemon= Evolves_from(pokemon)

if Evolves_from(pokemon) == None:                       # at the end of the chain, prints that there are no more preevolutions
    print(f"{pokemon} does not have a pre-evolution")



