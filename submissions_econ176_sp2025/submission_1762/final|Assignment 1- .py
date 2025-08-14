# we assign the url and obtain the api-call result into result
#    Note that result will be an object that contains many fields (not a simple string)
#

import requests
from math import radians, sin, cos, asin, sqrt

url = "http://api.open-notify.org/iss-now.json"   # this is sometimes called an "endpoint" ...
result = requests.get(url)

# if it succeeds, you should see <Response [200]>


#
# In this case, we know the result is a JSON file, and we can obtain it that way:

json_contents = result.json()      # needs to convert the text to a json dictionary...
print(f"json_contents is {json_contents}")     # Aha!  Let's re/introduce f-strings...

# Take a look... remember that a json object is a Python dictionary:


#
# Let's remind ourselves how dictionaries work:

lat = json_contents['iss_position']['latitude']
lat = float(lat)
lon = json_contents['iss_position']['longitude']
lon = float(lon)
print("lat: ", lat)
print("lon: ", lon)


#
# Let's make sure we "unpack the process" w/o AI
#
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

result = haversine(lat,lon,34.1007,117.706)
print(result)


#
# Then, let's compare with AI's result...
#

def haversine(lat1, long1, lat2, long2):
    """
    Calculate the great-circle distance (miles) between two lat/longs.
    """
    from math import radians, sin, cos, asin, sqrt
    long1, lat1, long2, lat2 = map(radians, [long1, lat1, long2, lat2])
    dlong = long2 - long1
    dlat = lat2 - lat1
    trig = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlong/2)**2
    radius = 3956  # earth radius in miles
    return 2 * radius * asin(sqrt(trig))

# 1. Get the current ISS position via the API
url = "http://api.open-notify.org/iss-now.json"
result = requests.get(url)
json_contents = result.json()

iss_lat = float(json_contents["iss_position"]["latitude"])
iss_long = float(json_contents["iss_position"]["longitude"])

# 2. Define Claremont's coordinates (e.g., Claremont, CA)
claremont_lat = 34.0967
claremont_long = -117.7198

# 3. Calculate the distance using haversine
distance_miles = haversine(iss_lat, iss_long, claremont_lat, claremont_long)

print(f"The ISS is currently {distance_miles:.2f} miles away from Claremont.")


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
d = astronauts   # a shorter variable for convenience..


# Remember:  astronauts will be a _dictionary_
note = """ here's yesterday evening's result - it _should_ be the same this morning!

{"people": [{"craft": "ISS", "name": "Oleg Kononenko"}, {"craft": "ISS", "name": "Nikolai Chub"},
{"craft": "ISS", "name": "Tracy Caldwell Dyson"}, {"craft": "ISS", "name": "Matthew Dominick"},
{"craft": "ISS", "name": "Michael Barratt"}, {"craft": "ISS", "name": "Jeanette Epps"},
{"craft": "ISS", "name": "Alexander Grebenkin"}, {"craft": "ISS", "name": "Butch Wilmore"},
{"craft": "ISS", "name": "Sunita Williams"}, {"craft": "Tiangong", "name": "Econ176_Participant_6 Guangsu"},
{"craft": "Tiangong", "name": "Econ176_Participant_6 Cong"}, {"craft": "Tiangong", "name": "Ye Guangfu"}], "number": 12, "message": "success"}
"""
print(d)


d['people']


#
# Try it - from a browser or from here...

import requests

url = "https://fvcjsw-5000.csb.app/econ176_mystery0?x=1&y=3"    # perhaps try from browser first!
result_ft = requests.get(url)
# print(result_ft)              # prints the status_code

d = result_ft.json()            # here are the _contents_



#
# A larger API call to the same CodeSandbox server

import requests

url = "https://fvcjsw-5000.csb.app/fintech"    # try this from your browser first!
result_ft = requests.get(url)
result_ft


#
# Let's view ... then parse and interpret!

d = result_ft.json()                  # try .text, as well...
print(f"The resulting data is {d}")


print(d['Initials'][10])


#
# see if you can extract only your initials from d
d['Initials']
d['Initials'][10]

# we're not finished yet! :)


#
# Let's request!   Just using the demo, for now:

import requests

url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&apikey=demo"    # demo version
result = requests.get(url)



#
# Let's view ... then parse and interpret!

d = result.json()
print(d)


#
# Let's look at all of the keys...

for k in d['Time Series (Daily)']:
    print(k)

# Aha! they are dates... let's create a function to compare two dates


#
# here is one way to make a list of all of the dates:

DATES = list(d['Time Series (Daily)'].keys())

# Notice, they're backwards!


#
# Let's flip the DATES around:
DATES.reverse()

# Yay!


# Oooh... Now let's see what's in each key (date)

val = float(d['Time Series (Daily)']['2025-01-21']['4. close'])  # Aha! it's a dictionary again!  We will need to index again!!



# A small function to get the closing price on a date (date) using data (dictionary) d
def get_closing(date, d):
    close = float(d['Time Series (Daily)'][date]['4. close'])
    return close


# A loop to find the minimum closing price
#

min_price = 10000000
min_key = "nothing"

for date in d['Time Series (Daily)']:
    closing =  get_closing(date, d)
    # print(f"date is {date} and closing is {closing}")
    if closing < min_price:
        min_price = closing
        min_price_date = date

print(f"min_price_date is {min_price_date} and {min_price = }")


#PART 1

import requests

# repeating api call with AppLovin (APP)
url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=APP&apikey=K8KHJR1NUZNEMFX1"    # demo version
result = requests.get(url)
d = result.json()

# making a list of closing prices
closingPrices = list()
for date in d['Time Series (Daily)']:
  closingPrices.append(d['Time Series (Daily)'][date]['4. close'])
closingPrices.reverse()

DATES = list(d['Time Series (Daily)'].keys())
DATES.reverse()

# finding max and min and corresponding dates
min_price = 10000000
max_price = 0
min_key = "nothing"


for date in d['Time Series (Daily)']:
    closing =  get_closing(date, d)
    # print(f"date is {date} and closing is {closing}")
    if closing < min_price:
        min_price = closing
        min_price_date = date
    if closing > max_price:
        max_price = closing
        max_price_date = date

print(f"min_price_date is {min_price_date} and {min_price = }")
print(f"max_price_date is {max_price_date} and {max_price = }")

# using AI to plot graph

import matplotlib.pyplot as plt
import seaborn as sns

plt.plot(DATES, closingPrices)
min_index = closingPrices.index(min(closingPrices))
max_index = closingPrices.index(max(closingPrices))

plt.plot(DATES[min_index], closingPrices[min_index], 'ro', markersize=10, label='Minimum')
plt.plot(DATES[max_index], closingPrices[max_index], 'go', markersize=10, label='Maximum')

plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('AppLovin (APP) Stock Prices')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.legend()
plt.tight_layout()  # Adjust layout to prevent labels from overlapping

plt.show()

# single share analysis
profit = max_price - min_price
print(f"buy at {min_price_date} for {min_price} and sell at {max_price_date} for {max_price} in order to make profit of {profit}")

# doing the same with another stock, this time: SMCI
url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=TOST&apikey=K8KHJR1NUZNEMFX1"    # demo version
result = requests.get(url)
d = result.json()
closingPrices = list()
for date in d['Time Series (Daily)']:
  closingPrices.append(d['Time Series (Daily)'][date]['4. close'])
closingPrices.reverse()
DATES = list(d['Time Series (Daily)'].keys())
DATES.reverse()
min_price = 10000000
max_price = 0
min_key = "nothing"
for date in d['Time Series (Daily)']:
    closing =  get_closing(date, d)
    # print(f"date is {date} and closing is {closing}")
    if closing < min_price:
        min_price = closing
        min_price_date = date
    if closing > max_price:
        max_price = closing
        max_price_date = date
import matplotlib.pyplot as plt
import seaborn as sns
plt.plot(DATES, closingPrices)
min_index = closingPrices.index(min(closingPrices))
max_index = closingPrices.index(max(closingPrices))
plt.plot(DATES[min_index], closingPrices[min_index], 'ro', markersize=10, label='Minimum')
plt.plot(DATES[max_index], closingPrices[max_index], 'go', markersize=10, label='Maximum')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Toast (TOST) Stock Prices')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.legend()
plt.show()
profit = max_price - min_price
print(f"buy at {min_price_date} for {min_price} and sell at {max_price_date} for {max_price} in order to make profit of {profit}")


# PART 2 - using week-end data, a new API call!
url = "https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=IBM&apikey=K8KHJR1NUZNEMFX1"    # demo version
result = requests.get(url)
d = result.json()

min_price = 10000000
max_price = 0
min_key = "nothing"

def get_closing_weekly(date, d):
    close = float(d['Weekly Time Series'][date]['4. close'])
    return close

total = 0
count = 0
for date in d['Weekly Time Series']:
    closing =  get_closing_weekly(date, d)
    total += closing
    count += 1
    # print(f"date is {date} and closing is {closing}")
    if closing < min_price:
        min_price = closing
        min_price_date = date
    if closing > max_price:
        max_price = closing
        max_price_date = date

print(f"min_price_date is {min_price_date} and {min_price = }")
print(f"max_price_date is {max_price_date} and {max_price = }")

avg_price = total // count
print(f"the average price is {avg_price}")


