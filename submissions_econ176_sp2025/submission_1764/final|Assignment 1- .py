# we assign the url and obtain the api-call result into result
#    Note that result will be an object that contains many fields (not a simple string)
#

import requests

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
print("lat: ", lat)


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


#
# Let's make sure we "unpack the process" w/o AI
#



#
# Then, let's compare with AI's result...
#


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


#
# see if you can extract only your initials from d

# we're not finished yet! :)


#
# Let's request!   Just using the demo, for now:

import requests

url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&apikey=demo"    # demo version
result = requests.get(url)



#
# Let's view ... then parse and interpret!

d = result.json()                       # probably _don't_ try .text here!
print(f"The resulting data's keys are {list(d.keys())}")


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

d['Time Series (Daily)']['2025-01-21']  # Aha! it's a dictionary again!  We will need to index again!!


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
        min_price = close
        min_price_date = date

print(f"min_price_date is {min_price_date} and {min_price = }")


# Request using my API key MTWVU7GBVGXVXNPB
# Microsoft Inc. Stock analysis
import requests

url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&apikey=MTWVU7GBVGXVXNPB"
result = requests.get(url)



#
# Let's view ... then parse and interpret!

d = result.json()
print(f"The resulting data's keys are {list(d.keys())}")


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
# Let's flip the DATES around to start in ascending order:
DATES.reverse()

# Yay!


# Oooh... Now let's see what's in each key (date)

d['Time Series (Daily)']['2025-01-21']  # Aha! it's a dictionary again!  We will need to index again!!


# A small function to get the closing price on a date (date) using data (dictionary) d
def get_closing(date, d):
    close = float(d['Time Series (Daily)'][date]['4. close'])
    return close


# Extract the closing price of the 100 Dates extracted.
close_price = []
for date in DATES:
    close = get_closing(date, d)
    close_price.append(close)

close_price


# Find Maximum and minimum closing prices and their respective dates
max_price = max(close_price)
min_price = min(close_price)

max_date = DATES[close_price.index(max_price)]
min_date = DATES[close_price.index(min_price)]

print(f"*Maximum closing price: {max_price} on {max_date}\n*Minimum closing price: {min_price} on {min_date}")


# Graph of closing prices over time
import matplotlib.pyplot as plt

plt.plot(DATES, close_price)
plt.xticks(ticks=DATES[::5], rotation=45)  # Minimize clustered dates on x-axis
plt.xlabel('Date', fontweight='bold')
plt.ylabel('Closing Price', fontweight='bold')
plt.title('Closing Price Over Time', fontweight='bold')
plt.show()


# Function for Single-Share Analysis
def single_share_analysis(prices):
    min_price, max_profit, buy_date, sell_date = float('inf'), 0, None, None

    for date, price in enumerate(prices):
        if price < min_price:
            min_price = price
            buy_date  = DATES[close_price.index(min_price)]

        if price - min_price > max_profit:
            max_profit = price - min_price
            sell_date = DATES[close_price.index(price)]

    return buy_date, sell_date, max_profit



# Check Single-Share Analysis:
buy_date, sell_date, max_profit = single_share_analysis(close_price)
print(f'The maximum profit achieved is: {max_profit}; by buying on: {buy_date} and selling on: {sell_date}')



# Find buy date and sell date indices
buy_idx = DATES.index(buy_date)
sell_idx = DATES.index(sell_date)


plt.plot(DATES, close_price)
plt.scatter(buy_idx, close_price[buy_idx], color='green', label='Buy Day', zorder=5)
plt.scatter(sell_idx, close_price[sell_idx], color='red', label='Sell Day', zorder=5)
plt.xlabel('Date', fontweight='bold')
plt.ylabel('Closing Price', fontweight='bold')
plt.xticks(ticks=DATES[::5], rotation=45)  # Minimize clustered dates on x-axis
plt.title('Closing Price Over Time', fontweight='bold')
plt.legend()
plt.show()


# Request using my API key MTWVU7GBVGXVXNPB
# Apple Inc. Stock analysis
import requests

url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey=MTWVU7GBVGXVXNPB"
result = requests.get(url)



#
# Let's view ... then parse and interpret!

d = result.json()
print(f"The resulting data's keys are {list(d.keys())}")


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
# Let's flip the DATES around to start in ascending order:
DATES.reverse()

# Yay!


# Oooh... Now let's see what's in each key (date)

d['Time Series (Daily)']['2025-01-21']  # Aha! it's a dictionary again!  We will need to index again!!


# A small function to get the closing price on a date (date) using data (dictionary) d
def get_closing(date, d):
    close = float(d['Time Series (Daily)'][date]['4. close'])
    return close


# Extract the closing price of the 100 Dates extracted.
close_price = []
for date in DATES:
    close = get_closing(date, d)
    close_price.append(close)

close_price


# Find Maximum and minimum closing prices and their respective dates
max_price = max(close_price)
min_price = min(close_price)

max_date = DATES[close_price.index(max_price)]
min_date = DATES[close_price.index(min_price)]

print(f"*Maximum closing price: {max_price} on {max_date}\n*Minimum closing price: {min_price} on {min_date}")


# Graph of closing prices over time
import matplotlib.pyplot as plt

plt.plot(DATES, close_price)
plt.xlabel('Date', fontweight='bold')
plt.ylabel('Closing Price', fontweight='bold')
plt.title('Closing Price Over Time', fontweight='bold')
plt.show()


# Function for Single-Share Analysis
def single_share_analysis(prices):
    min_price, max_profit, buy_date, sell_date = float('inf'), 0, None, None

    for date, price in enumerate(prices):
        if price < min_price:
            min_price = price
            buy_date  = DATES[close_price.index(min_price)]

        if price - min_price > max_profit:
            max_profit = price - min_price
            sell_date = DATES[close_price.index(price)]

    return buy_date, sell_date, max_profit



# Check Single-Share Analysis:
buy_date, sell_date, max_profit = single_share_analysis(close_price)
print(f'The maximum profit achieved is: {max_profit}; by buying on: {buy_date} and selling on: {sell_date}')



# Find buy date and sell date indices
buy_idx = DATES.index(buy_date)
sell_idx = DATES.index(sell_date)


plt.plot(DATES, close_price)
plt.scatter(buy_idx, close_price[buy_idx], color='green', label='Buy Day', zorder=5)
plt.scatter(sell_idx, close_price[sell_idx], color='red', label='Sell Day', zorder=5)
plt.xlabel('Date', fontweight='bold')
plt.ylabel('Closing Price', fontweight='bold')
plt.xticks(ticks=DATES[::5], rotation=45)  # Minimize clustered dates on x-axis
plt.title('Closing Price Over Time', fontweight='bold')
plt.legend()
plt.show()


