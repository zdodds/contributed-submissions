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


import requests

url = "http://api.open-notify.org/iss-now.json"   # this is sometimes called an "endpoint" ...
result = requests.get(url)

json_contents = result.json()      # needs to convert the text to a json dictionary...
print(f"json_contents is {json_contents}")

lat = json_contents['iss_position']['latitude']
lat = float(lat)
print("lat: ", lat)


lon = json_contents['iss_position']['longitude']
lon = float(lon)
print("lon: ", lon)

claremont_lat = 34.0967
claremont_lon = 117.7198
result = haversine(lat, lon, claremont_lat, claremont_lon)

print(result)


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


print(len(d))


print(type(d['people']))


count = 0
for entry in d['people']:
  if entry['craft'] == "ISS":
    count += 1
print(count)


print(d['people'][-4]['name'])


print(d['people'][0]['name'][7:9])


#
# Try it - from a browser or from here...

import requests

url = "https://fvcjsw-5000.csb.app/econ176_mystery0?x=1&y=3"    # perhaps try from browser first!
result_ft = requests.get(url)
# print(result_ft)              # prints the status_code

#d = result_ft.json()            # here are the _contents_
#d


#
# A larger API call to the same CodeSandbox server

import requests

url = "https://fvcjsw-5000.csb.app/fintech"    # try this from your browser first!
result_ft = requests.get(url)
result_ft


#
# Let's view ... then parse and interpret!

#d = result_ft.json()                  # try .text, as well...
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
    close =  get_closing(date, d)
    # print(f"date is {date} and closing is {closing}")
    if close < min_price:
        min_price = close
        min_price_date = date

print(f"min_price_date is {min_price_date} and {min_price = }")


import requests


url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&apikey=147LGXMNKZFU94BN"
result = requests.get(url)
d = result.json()

url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=TSLA&apikey=147LGXMNKZFU94BN"
result2 = requests.get(url)
d2 = result2.json()



import matplotlib.pyplot as plt
import pandas as pd

def get_closing(date, d):
    close = float(d['Time Series (Daily)'][date]['4. close'])
    return close

prices = []
for date in d['Time Series (Daily)']:
  prices.append((get_closing(date, d), date))

prices2 = []
for date in d2['Time Series (Daily)']:
  prices2.append((get_closing(date, d2), date))

prices= sorted(prices, key=lambda x: x[1])
prices2= sorted(prices2, key=lambda x: x[1])
dates = [pair[1] for pair in prices]
closing_prices = [pair[0] for pair in prices]

max_price = max(closing_prices)
min_price = min(closing_prices)

dates = pd.to_datetime(dates)

max_date = dates[closing_prices.index(max_price)]
min_date = dates[closing_prices.index(min_price)]

print(f"Maximum price: {max_price} on {max_date.strftime('%Y-%m-%d')}")
print(f"Minimum price: {min_price} on {min_date.strftime('%Y-%m-%d')}")

plt.figure(figsize=(10, 5))
plt.plot(dates, closing_prices, label='Closing Prices', color='b')

plt.scatter(max_date, max_price, color='r', label=f'Max: {max_price} on {max_date.strftime("%Y-%m-%d")}', zorder=5)
plt.scatter(min_date, min_price, color='g', label=f'Min: {min_price} on {min_date.strftime("%Y-%m-%d")}', zorder=5)

plt.title('Closing Prices with Max and Min Highlighted')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
plt.show()



#Calculating buy day and sell day

left, right = 0, 1
profit = 0
buyDay = ""
sellDay = ""
buyDayPrice = 0
sellDayPrice = 0
while right < len(prices):
    if prices[left][0] < prices[right][0]:
        current_profit = prices[right][0] - prices[left][0]
        if current_profit > profit:
          profit = current_profit
          buyDay = prices[left][1]
          sellDay = prices[right][1]
          buyDayPrice = prices[left][0]
          sellDayPrice = prices[right][0]
    else:
        left = right
    right += 1
print(f"Buy Day: {buyDay} with a Price of {buyDayPrice}")
print(f"Sell Day: {sellDay} with a Price of {sellDayPrice}")
print(f"Profit:  {profit}")





plt.figure(figsize=(10, 5))
plt.plot(dates, closing_prices, label='Closing Prices', color='b')

buyDay_dt = pd.to_datetime(buyDay)
sellDay_dt = pd.to_datetime(sellDay)

plt.scatter(max_date, max_price, color='r', label=f'Max: {max_price} on {max_date.strftime("%Y-%m-%d")}', zorder=5)
plt.scatter(min_date, min_price, color='g', label=f'Min: {min_price} on {min_date.strftime("%Y-%m-%d")}', zorder=5)
plt.scatter(sellDay_dt, sellDayPrice, color='b', marker='o', s=100, label=f'Sell: {max_price} on {max_date.strftime("%Y-%m-%d")}')
plt.scatter(buyDay_dt, buyDayPrice, color='y', marker='o', s=100, label=f'Buy: {min_price} on {min_date.strftime("%Y-%m-%d")}')


plt.title('Closing Prices with Max and Min Highlighted')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
plt.show()


#Exploring with other API
import requests


url = "https://www.alphavantage.co/query?function=INSIDER_TRANSACTIONS&symbol=GOOG&apikey=147LGXMNKZFU94BN"    # demo version
result = requests.get(url)
d = result.json()

print(d)


#API Exploration continued
import heapq
mp = {}
for entry in d['data']:
  if entry['executive'] not in mp:
    mp[entry['executive']] = 1
  else:
    mp[entry['executive']] += 1


top_5_items = heapq.nlargest(5, mp.items(), key=lambda x: x[1])

top_5_dict = dict(top_5_items)

print(top_5_dict)


#Extra Credit

import yfinance as yf
from scipy.stats import pearsonr
import pandas as pd


# Function to compute Pearson correlation coefficient between two stock prices
def compute_pearson_correlation(stock_data_1, stock_data_2):
    # Align the data based on the common dates
    merged_data = pd.merge(stock_data_1, stock_data_2, on='Date', how='inner')
    correlation, _ = pearsonr(merged_data['Price_x'], merged_data['Price_y'])
    return correlation

df1 = pd.DataFrame(prices, columns=["Price", "Date"])
df2 = pd.DataFrame(prices2, columns=["Price", "Date"])

# Compute correlation
correlation = compute_pearson_correlation(df1, df2)
print(f"Pearson Correlation Coefficient between MSFT and TSLA: {correlation}")


