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



import requests
from math import radians, sin, cos, sqrt, asin

url = "http://api.open-notify.org/iss-now.json"
result = requests.get(url)
json_contents = result.json()

# Extracting latitude and longitude of the ISS
iss_lat = float(json_contents['iss_position']['latitude'])
iss_long = float(json_contents['iss_position']['longitude'])

print(f"ISS Current Location: Latitude={iss_lat}, Longitude={iss_long}")

# Coordinates of Claremont:
claremont_lat = 34.0967
claremont_long = -117.7198

lat1, long1, lat2, long2 = map(radians, [iss_lat, iss_long, claremont_lat, claremont_long])
dlat = lat2 - lat1
dlong = long2 - long1
a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlong/2)**2
c = 2 * asin(sqrt(a))
radius = 3956
distance_manual = radius * c

print(f"Manual Distance Calculation (without AI): {distance_manual:.2f} miles")



def haversine(lat1, long1, lat2, long2):
    from math import radians, sin, cos, sqrt, asin
    long1, lat1, long2, lat2 = map(radians, [long1, lat1, long2, lat2])
    dlong = long2 - long1
    dlat = lat2 - lat1
    trig = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlong/2)**2
    radius = 3956
    return radius * 2 * asin(sqrt(trig))

distance_ai = haversine(iss_lat, iss_long, claremont_lat, claremont_long)

print(f"Distance Calculation (using AI/haversine function): {distance_ai:.2f} miles")

# Comparing both results
if abs(distance_manual - distance_ai) < 0.01:
    print("Both calculations are similar!")
else:
    print("There is a difference.")



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

d = result.json()
print(d)


#
# Let's look at all of the keys...

for k in d['Time Series (Daily)']:
    print(k)
# Let's look at all of the keys...

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

print(f"min_price_date is {min_price_date} and min_price = {min_price}")



import requests
import matplotlib.pyplot as plt

def fetch_stock_data(symbol, api_key="demo"):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"
    result = requests.get(url)
    return result.json()

def get_closing(date, data):
    return float(data['Time Series (Daily)'][date]['4. close'])

def analyze_stock(data):
    time_series = data['Time Series (Daily)']
    dates = list(time_series.keys())
    closing_prices = [get_closing(date, data) for date in dates]

    max_price = max(closing_prices)
    min_price = min(closing_prices)
    max_date = dates[closing_prices.index(max_price)]
    min_date = dates[closing_prices.index(min_price)]

    max_profit = 0
    buy_date = sell_date = ""
    for i in range(len(closing_prices)):
        for j in range(i+1, len(closing_prices)):
            profit = closing_prices[j] - closing_prices[i]
            if profit > max_profit:
                max_profit = profit
                buy_date = dates[i]
                sell_date = dates[j]

    return dates, closing_prices, max_price, max_date, min_price, min_date, buy_date, sell_date, max_profit

def plot_stock(dates, prices, max_price, max_date, min_price, min_date, buy_date, sell_date):
    dates.reverse()
    prices.reverse()
    plt.figure(figsize=(12, 6))
    plt.plot(dates, prices, label='Closing Prices')
    plt.scatter([max_date], [max_price], color='red', label=f'Max: ${max_price}', marker='o')
    plt.scatter([min_date], [min_price], color='green', label=f'Min: ${min_price}', marker='o')
    plt.scatter([buy_date, sell_date], [prices[dates.index(buy_date)], prices[dates.index(sell_date)]], color='orange', label='Buy/Sell Points', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title('Stock Prices with Max/Min and Buy/Sell Points')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.show()

symbol = input("Enter stock ticker symbol (e.g., AAPL, MSFT, IBM): ").upper()
data = fetch_stock_data(symbol)
dates, prices, max_price, max_date, min_price, min_date, buy_date, sell_date, max_profit = analyze_stock(data)

print(f"\nMax Price: ${max_price} on {max_date}")
print(f"Min Price: ${min_price} on {min_date}")
print(f"Best Buy Date: {buy_date}, Sell Date: {sell_date}, Max Profit: ${max_profit:.2f}")

plot_stock(dates, prices, max_price, max_date, min_price, min_date, buy_date, sell_date)



