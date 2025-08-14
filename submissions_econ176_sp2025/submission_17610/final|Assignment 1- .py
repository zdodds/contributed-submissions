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


#
# Let's make sure we "unpack the process" w/o AI
#
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


url = "http://api.open-notify.org/iss-now.json"   # this is sometimes called an "endpoint" ...
result = requests.get(url)
json_contents = result.json()


iss_lat = float(json_contents['iss_position']['latitude'])
iss_long = float(json_contents['iss_position']['longitude'])

my_lat = 34.1063626
my_long = -117.7111075


haversine(iss_lat, iss_long, my_lat, my_long)


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

url = "https://fvcjsw-5000.csb.app/econ176_mystery0?x=0&y=0"    # perhaps try from browser first!
result_ft = requests.get(url)
# print(result_ft)              # prints the status_code

d = result_ft.json()            # here are the _contents_

# multiplication


#
# Try it - from a browser or from here...

import requests

url = "https://fvcjsw-5000.csb.app/econ176_mystery1?x=15&y=4"    # perhaps try from browser first!
result_ft = requests.get(url)
# print(result_ft)              # prints the status_code

d = result_ft.json()            # here are the _contents_


# if x odd, then y*2, if x even, then y


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
d['Initials'][-17]

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
        min_price = closing
        min_price_date = date

print(f"min_price_date is {min_price_date} and {min_price = }")


api_key = ''


from google.colab import userdata
api_key = userdata.get('API_KEY')


import requests


def single_share_analysis(symbol, api_key):


    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"    # demo version
    result = requests.get(url)
    if result.status_code != 200:
        raise Exception(f"result.status_code = {result.status_code}")
    d = result.json()



    # prompt: find the maximum and the minimum
    # also find the date of the max and the date of the min
    # be sure to print those out...


    DATES = list(d['Time Series (Daily)'].keys())
    DATES.reverse()

    # programmatically extract the 100 prices (let's use the closing price)
    # create a list with them

    prices = []

    for date in d['Time Series (Daily)']:
        close = float(d['Time Series (Daily)'][date]['4. close'])
        prices.append((date, close))


    # Find the maximum and minimum closing prices and their corresponding dates
    max_price = -1
    min_price = float('inf')
    max_date = None
    min_date = None

    for date, price in prices:
        if price > max_price:
            max_price = price
            max_date = date
        if price < min_price:
            min_price = price
            min_date = date

    print(f"Maximum closing price: {max_price} on {max_date}")
    print(f"Minimum closing price: {min_price} on {min_date}")

    # Single-share analysis: Find the buy day and sell day that maximize profit
    max_profit = 0
    buy_date = None
    sell_date = None

    for i in range(len(prices)):
        for j in range(i, len(prices)):
            profit = prices[j][1] - prices[i][1]
            if profit > max_profit:
                max_profit = profit
                buy_date = prices[i][0]
                sell_date = prices[j][0]

    print(f"Maximum profit: {max_profit}")
    print(f"Buy date: {buy_date}")
    print(f"Sell date: {sell_date}")

    # Graphing (using matplotlib)
    import matplotlib.pyplot as plt

    dates = [date for date, price in prices]
    prices_only = [price for date, price in prices]

    plt.figure(figsize=(12, 6))
    plt.plot(dates, prices_only)
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.title(f"{symbol} Stock Prices")
    plt.xticks(rotation=45)

    # Highlight the maximum and minimum points
    plt.scatter(max_date, max_price, color='green', label='Maximum')
    plt.scatter(min_date, min_price, color='red', label='Minimum')
    plt.scatter(buy_date, prices_only[dates.index(buy_date)], color='blue', label='Buy')
    plt.scatter(sell_date, prices_only[dates.index(sell_date)], color='orange', label='Sell')


    plt.legend()
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.show()



single_share_analysis('GOOG', api_key)


single_share_analysis('FYBR', api_key)


# Import necessary libraries
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



def get_stock_data(symbol, api_key, start_date=None):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol={symbol}&apikey={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error: status code {response.status_code}")
    data = response.json()

    # Convert the JSON data into a DataFrame
    time_series = data.get("Time Series (Daily)", {})
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Convert closing price to float
    df['close'] = df['4. close'].astype(float)

    # Filter by start_date if provided
    if start_date:
        start_date = pd.to_datetime(start_date)
        df = df[df.index >= start_date]

    return df[['close']]


# Example usage:
symbol = "NVDA"
start_date = "2022-04-01"
stock_df = get_stock_data(symbol, api_key, start_date)
# Adjust for stock split on 2024-06-10
split_date = pd.to_datetime("2024-06-10")
stock_df.loc[stock_df.index < split_date, 'close'] /= 10
stock_df



# V2

import requests
from datetime import datetime, timedelta
import json
import os

# Define a cache file name
CACHE_FILE = 'sentiment_cache.json'

# Load existing cache if available, otherwise create an empty cache dictionary
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r') as f:
        cache = json.load(f)
else:
    cache = {}

def get_sentiment_data_segment(symbol, time_from, time_to, api_key, sort='LATEST', limit="1000"):
    """
    Fetch sentiment data for a single time segment from time_from to time_to.
    Uses caching to avoid repeated API calls.
    """
    # Build a unique cache key based on the parameters
    cache_key = f"{symbol}_{time_from}_{time_to}_{sort}_{limit}"
    if cache_key in cache:
        print("Using cached data for:", cache_key)
        return cache[cache_key]

    # Build the URL using the provided documentation attributes:
    # - time_from and time_to in the format YYYYMMDDTHHMM
    # - sort parameter (default 'LATEST')
    # - limit parameter (default "1000")
    url = (f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}"
           f"&time_from={time_from}&time_to={time_to}&sort={sort}&limit={limit}&apikey={api_key}")
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error: status code {response.status_code}")

    sentiment_data = response.json()

    # Cache the result for future runs
    cache[cache_key] = sentiment_data
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

    return sentiment_data

def get_sentiment_data(symbol, start_time, end_time, api_key, segment_hours=24, sort='LATEST', limit="1000"):
    """
    Loop over the time range from start_time to end_time in segments,
    fetching and appending sentiment data from each segment.

    Parameters:
      - symbol: Ticker symbol for filtering news
      - start_time: Start of the time range (YYYYMMDDTHHMM format)
      - end_time: End of the time range (YYYYMMDDTHHMM format)
      - api_key: Your API key for Alphavantage
      - segment_hours: The length of each segment in hours (default: 24)
      - sort: Sort order for the API (default: 'LATEST')
      - limit: Maximum number of articles per segment (default: "1000")

    Returns:
      A dictionary with combined sentiment data.
    """
    # Convert the provided start and end times into datetime objects
    start_dt = datetime.strptime(start_time, "%Y%m%dT%H%M")
    end_dt = datetime.strptime(end_time, "%Y%m%dT%H%M")
    all_feed = []

    current_from = start_dt
    while current_from < end_dt:
        current_to = current_from + timedelta(hours=segment_hours)
        if current_to > end_dt:
            current_to = end_dt

        # Convert back to string format required by the API
        time_from_str = current_from.strftime("%Y%m%dT%H%M")
        time_to_str = current_to.strftime("%Y%m%dT%H%M")
        print(f"Fetching sentiment data from {time_from_str} to {time_to_str}")

        segment_data = get_sentiment_data_segment(symbol, time_from_str, time_to_str, api_key, sort, limit)
        if "feed" in segment_data:
            all_feed.extend(segment_data["feed"])
        else:
            print("No feed data in segment:", time_from_str, "to", time_to_str)

        current_from = current_to

    # Combine the data from all segments into a single dictionary.
    # We also include the sentiment score and relevance definitions from the last segment fetched.
    combined_data = {
        "items": len(all_feed),
        "sentiment_score_definition": segment_data.get("sentiment_score_definition", ""),
        "relevance_score_definition": segment_data.get("relevance_score_definition", ""),
        "feed": all_feed
    }
    return combined_data

# Example usage:
# Define your API key, symbol, and desired time range:
symbol = "NVDA"
start_time = "20220410T0130"  # Example start time (YYYYMMDDTHHMM)
end_time = "20250214T2359"    # Example end time (YYYYMMDDTHHMM)

sentiment_data = get_sentiment_data(symbol, start_time, end_time, api_key, segment_hours=24)
print("Combined sentiment data items:", sentiment_data["items"])



import pandas as pd
from datetime import datetime

# Inspect main keys and number of feed items
print("Main keys in sentiment_data:", sentiment_data.keys())
print("Total number of feed items:", sentiment_data["items"])

# Convert the feed data into a DataFrame for easier analysis
feed_df = pd.DataFrame(sentiment_data["feed"])
print("\nFeed DataFrame head:")
print(feed_df.head())

# Convert 'time_published' to datetime format (extracting date portion)
def parse_time_published(time_str):
    # Assuming format "YYYYMMDDThhmmss"
    return datetime.strptime(time_str, "%Y%m%dT%H%M%S")

# Apply the conversion if the column exists
if 'time_published' in feed_df.columns:
    feed_df['datetime'] = feed_df['time_published'].apply(parse_time_published)
    feed_df['date'] = feed_df['datetime'].dt.date

    # Check the time range available in the feed data
    min_date = feed_df['datetime'].min()
    max_date = feed_df['datetime'].max()
    print("\nTime range in sentiment data:")
    print(f"From: {min_date} To: {max_date}")

    # Show basic sentiment statistics
    if 'overall_sentiment_score' in feed_df.columns:
        print("\nSentiment score stats:")
        print(feed_df['overall_sentiment_score'].describe())
else:
    print("No 'time_published' column found in the feed data.")

# At this point, feed_df should have a 'date' column that you can use to merge with the stock data.



# 7

# Aggregate sentiment data by date: average overall sentiment score per day
daily_sentiment = feed_df.groupby('date')['overall_sentiment_score'].mean().reset_index()
daily_sentiment.rename(columns={'overall_sentiment_score': 'daily_sentiment'}, inplace=True)
print("Daily sentiment head:")
print(daily_sentiment.head())

# Check time range of aggregated sentiment data
print("Aggregated sentiment time range: {} to {}".format(daily_sentiment['date'].min(), daily_sentiment['date'].max()))



# 8: Merge Stock Data with Daily Sentiment Data

# Convert daily_sentiment 'date' column to datetime (if not already)
daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])

# Reset index for stock_df and ensure the 'date' column is datetime
stock_df_reset = stock_df.reset_index().rename(columns={'index': 'date'})
stock_df_reset['date'] = pd.to_datetime(stock_df_reset['date'])

# Merge on the 'date' column using a left join
merged_df = pd.merge(stock_df_reset, daily_sentiment, on='date', how='left')

# Fill missing sentiment values with forward fill, then backward fill as a safeguard
merged_df['daily_sentiment'] = merged_df['daily_sentiment'].ffill().bfill()

print("Merged DataFrame head after subsetting:")
print(merged_df.head())



# 8.5 get excess return

symbol = "SPY"
start_date = "2022-04-01"
index_df = get_stock_data(symbol, api_key, start_date)
# Reset index for index_df and ensure the 'date' column is datetime
index_df_reset = index_df.reset_index().rename(columns={'index': 'date'})
index_df_reset['date'] = pd.to_datetime(index_df_reset['date'])
index_df




# 9 v2
# 9: Compute Daily Returns, Excess Returns, and Sentiment Derivative

# --- For the stock ---
# Ensure merged_df is sorted by date and compute the stock's daily return
merged_df = merged_df.sort_values('date')
merged_df['return'] = merged_df['close'].pct_change()
merged_df = merged_df.dropna(subset=['return'])  # Remove first row with NaN return

print("Merged DataFrame with stock returns:")
print(merged_df.head())

# --- For the index (SPY) ---
# Ensure index_df_reset is sorted by date and compute SPY's daily return
index_df_reset = index_df_reset.sort_values('date')
index_df_reset['index_return'] = index_df_reset['close'].pct_change()
index_df_reset = index_df_reset.dropna(subset=['index_return'])

print("\nIndex DataFrame with returns:")
print(index_df_reset.head())

# --- Merge stock data with index data on 'date' ---
# We use an inner join to keep only dates that exist in both datasets.
merged_all = pd.merge(merged_df, index_df_reset[['date', 'index_return']], on='date', how='inner')

# --- Calculate Excess Return ---
# Excess Return = Stock Return - Index Return
merged_all['excess_return'] = merged_all['return'] - merged_all['index_return']

# --- Calculate the derivative (change) of sentiment over time ---
# This is simply the daily difference in the aggregated sentiment value.
merged_all = merged_all.sort_values('date')
merged_all['sentiment_change'] = merged_all['daily_sentiment'].diff()
merged_all = merged_all.dropna(subset=['sentiment_change'])  # Drop first row with NaN sentiment change

print("\nMerged DataFrame with Excess Returns and Sentiment Change:")
print(merged_all.head())

merged_all


# explore 10 v2

# 10: Exploratory Analysis of Relationships Between Variables

import seaborn as sns
import matplotlib.pyplot as plt

# Select the variables of interest
vars_of_interest = merged_all[['daily_sentiment', 'sentiment_change', 'return', 'index_return', 'excess_return']]

# Print the correlation matrix
print("Correlation matrix:")
print(vars_of_interest.corr())

# Create a pairplot to visualize relationships between variables
sns.pairplot(vars_of_interest)
plt.suptitle("Pairplot of Sentiment and Return Variables", y=1.02)
plt.show()

# Scatter plot: sentiment_change vs. excess_return
plt.figure(figsize=(8, 6))
plt.scatter(merged_all['sentiment_change'], merged_all['excess_return'], alpha=0.6)
plt.xlabel("Sentiment Change")
plt.ylabel("Excess Return")
plt.title("Scatter Plot: Sentiment Change vs. Excess Return")
plt.show()

# Scatter plot: daily_sentiment vs. excess_return
plt.figure(figsize=(8, 6))
plt.scatter(merged_all['daily_sentiment'], merged_all['excess_return'], alpha=0.6, color='green')
plt.xlabel("Daily Sentiment")
plt.ylabel("Excess Return")
plt.title("Scatter Plot: Daily Sentiment vs. Excess Return")
plt.show()



# 11 v2: Regression Analysis: Daily Sentiment vs. Excess Return

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Prepare the features (X) and target (y)
X = merged_all[['daily_sentiment']].values  # Predictor: daily sentiment
y = merged_all['excess_return'].values      # Target: excess return

# Build and fit the regression model
reg_model = LinearRegression()
reg_model.fit(X, y)

# Make predictions
y_pred = reg_model.predict(X)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print("Coefficient:", reg_model.coef_[0])
print("Intercept:", reg_model.intercept_)

# Plot scatter plot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(merged_all['daily_sentiment'], y, color='blue', alpha=0.6, label='Actual Excess Return')
plt.plot(merged_all['daily_sentiment'], y_pred, color='red', label='Fitted Line')
plt.xlabel("Daily Sentiment")
plt.ylabel("Excess Return")
plt.title("Regression: Daily Sentiment vs. Excess Return")
plt.legend()
plt.show()



# 9: Compute Daily Returns

# If daily returns haven't been computed, calculate percentage change for 'close'
# If returns already exist in stock_df, ensure they are included. Otherwise, calculate here:
merged_df = merged_df.sort_values('date')  # Ensure data is sorted by date
merged_df['return'] = merged_df['close'].pct_change()
merged_df = merged_df.dropna(subset=['return'])  # Remove the first row with NaN return

# Display the first few rows to confirm the 'return' column
print("Merged DataFrame with returns:")
merged_df




# 10?: Compute Cumulative Values and Inspect Sample Data

# Ensure data is sorted by date
merged_df = merged_df.sort_values('date')

# Compute cumulative return (starting with an initial value of 1)
merged_df['cum_return'] = (1 + merged_df['return']).cumprod()

# Compute cumulative sentiment as the cumulative sum of daily sentiment
merged_df['cum_sentiment'] = merged_df['daily_sentiment'].cumsum()

# Output sample data points
print("First 10 rows:")
print(merged_df[['date', 'close', 'return', 'cum_return', 'daily_sentiment', 'cum_sentiment']].head(10))

print("\nLast 10 rows:")
print(merged_df[['date', 'close', 'return', 'cum_return', 'daily_sentiment', 'cum_sentiment']].tail(10))



# 10: Dual-Axis Plot for Cumulative Return and Cumulative Sentiment

import matplotlib.pyplot as plt

# Create a dual-axis plot
fig, ax1 = plt.subplots(figsize=(12,6))

# Plot cumulative return on the left y-axis
color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Cumulative Return', color=color)
ax1.plot(merged_df['date'], merged_df['cum_return'], color=color, label='Cumulative Return')
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis for cumulative sentiment
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Cumulative Sentiment', color=color)
ax2.plot(merged_df['date'], merged_df['cum_sentiment'], color=color, label='Cumulative Sentiment')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title("Cumulative Return and Cumulative Sentiment Over Time")
plt.xticks(rotation=45)
plt.show()

# Output sample cumulative values for inspection
print("Sample cumulative values:")
print(merged_df[['date', 'cum_return', 'cum_sentiment']].tail(10))




# Prepare the features (X) and target (y) for regression
X = merged_df[['daily_sentiment']].values  # Predictor: daily sentiment
y = merged_df['return'].values              # Target: daily stock return

# Build the regression model using LinearRegression from scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

# Plot the actual vs predicted returns over time
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(merged_df['date'], y, color='blue', label='Actual Returns', alpha=0.6)
plt.plot(merged_df['date'], y_pred, color='red', label='Predicted Returns')
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.title("Regression: Daily Returns vs. Daily Sentiment")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import json
import os




def get_stock_data(symbol, api_key, start_date=None):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol={symbol}&apikey={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error: status code {response.status_code}")
    data = response.json()

    # Convert the JSON data into a DataFrame
    time_series = data.get("Time Series (Daily)", {})
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Convert closing price to float
    df['close'] = df['4. close'].astype(float)

    # Filter by start_date if provided
    if start_date:
        start_date = pd.to_datetime(start_date)
        df = df[df.index >= start_date]

    return df[['close']]



# Example usage for stock data:
symbol = "NVDA"
start_date = "2022-04-01"
stock_df = get_stock_data(symbol, api_key, start_date)

# Adjust for stock split on 2024-06-10
split_date = pd.to_datetime("2024-06-10")
stock_df.loc[stock_df.index < split_date, 'close'] /= 10

stock_df



# Define cache file for sentiment data
CACHE_FILE = 'sentiment_cache.json'

# Load cache if available; otherwise, create an empty cache
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r') as f:
        cache = json.load(f)
else:
    cache = {}

def get_sentiment_data_segment(symbol, time_from, time_to, api_key, sort='LATEST', limit="1000"):
    """
    Fetch sentiment data for a single time segment from time_from to time_to.
    Uses caching to avoid repeated API calls.
    """
    cache_key = f"{symbol}_{time_from}_{time_to}_{sort}_{limit}"
    if cache_key in cache:
        print("Using cached data for:", cache_key)
        return cache[cache_key]

    url = (f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}"
           f"&time_from={time_from}&time_to={time_to}&sort={sort}&limit={limit}&apikey={api_key}")
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error: status code {response.status_code}")

    sentiment_data = response.json()

    cache[cache_key] = sentiment_data
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

    return sentiment_data

def get_sentiment_data(symbol, start_time, end_time, api_key, segment_hours=24, sort='LATEST', limit="1000"):
    """
    Loop over the time range from start_time to end_time in segments,
    fetching and appending sentiment data from each segment.
    """
    start_dt = datetime.strptime(start_time, "%Y%m%dT%H%M")
    end_dt = datetime.strptime(end_time, "%Y%m%dT%H%M")
    all_feed = []

    current_from = start_dt
    while current_from < end_dt:
        current_to = current_from + timedelta(hours=segment_hours)
        if current_to > end_dt:
            current_to = end_dt

        time_from_str = current_from.strftime("%Y%m%dT%H%M")
        time_to_str = current_to.strftime("%Y%m%dT%H%M")
        print(f"Fetching sentiment data from {time_from_str} to {time_to_str}")

        segment_data = get_sentiment_data_segment(symbol, time_from_str, time_to_str, api_key, sort, limit)
        if "feed" in segment_data:
            all_feed.extend(segment_data["feed"])
        else:
            print("No feed data in segment:", time_from_str, "to", time_to_str)

        current_from = current_to

    combined_data = {
        "items": len(all_feed),
        "sentiment_score_definition": segment_data.get("sentiment_score_definition", ""),
        "relevance_score_definition": segment_data.get("relevance_score_definition", ""),
        "feed": all_feed
    }
    return combined_data



# Example usage for sentiment data:
symbol = "NVDA"
start_time = "20220410T0130"  # Format: YYYYMMDDTHHMM
end_time = "20250214T2359"    # Format: YYYYMMDDTHHMM

sentiment_data = get_sentiment_data(symbol, start_time, end_time, api_key, segment_hours=24)
print("Combined sentiment data items:", sentiment_data["items"])



# Cell 7 – Aggregate and Merge Sentiment with Stock Data

if 'time_published' in feed_df.columns:
    feed_df['time_published'] = pd.to_datetime(feed_df['time_published'])
    feed_df['date'] = feed_df['time_published'].dt.date
else:
    raise KeyError("Column 'time_published' not found in feed_df.")

# Aggregate sentiment data by date: average overall sentiment score per day
daily_sentiment = feed_df.groupby('date')['overall_sentiment_score'].mean().reset_index()
daily_sentiment.rename(columns={'overall_sentiment_score': 'daily_sentiment'}, inplace=True)
print("Daily sentiment head:")
print(daily_sentiment.head())

print("Aggregated sentiment time range: {} to {}".format(daily_sentiment['date'].min(), daily_sentiment['date'].max()))

# Merge with stock data:
# Convert stock_df index to a date column matching the format in daily_sentiment
stock_df_reset = stock_df.reset_index().rename(columns={'index': 'date'})
stock_df_reset['date'] = pd.to_datetime(stock_df_reset['date']).dt.date

merged_df = pd.merge(stock_df_reset, daily_sentiment, on='date', how='left')

# Fill missing sentiment values with forward fill, then backward fill as a safeguard
merged_df['daily_sentiment'] = merged_df['daily_sentiment'].ffill().bfill()

print("Merged DataFrame head after subsetting:")
print(merged_df.head())



# 8

# Get index data for symbol "SPY"
symbol_index = "SPY"
index_df = get_stock_data(symbol_index, api_key, start_date)
index_df_reset = index_df.reset_index().rename(columns={'index': 'date'})
index_df_reset['date'] = pd.to_datetime(index_df_reset['date'])

# Display the index data DataFrame
index_df



# 9

# Sort merged data by date and compute percentage change in 'close'
merged_df = merged_df.sort_values('date')
merged_df['return'] = merged_df['close'].pct_change()
merged_df = merged_df.dropna(subset=['return'])

print("Merged DataFrame with returns:")
print(merged_df.head())



# Cell 10 – Perform Regression Analysis and Plot Results with Enhanced Visualization

# Prepare features (X) and target (y)
X = merged_df[['daily_sentiment']].values  # Predictor: daily sentiment
y = merged_df['return'].values              # Target: daily stock return

# Build and train the regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

# Create a visually appealing plot with a refined theme
plt.style.use('dark_background')  # Set a nice theme for the plot
plt.figure(figsize=(12, 7))

# Plot actual returns as a scatter plot
plt.scatter(merged_df['date'], y, color='mediumblue', label='Actual Returns', alpha=0.7, s=60)

# Plot predicted returns as a line plot
plt.plot(merged_df['date'], y_pred, color='crimson', linewidth=2, label='Predicted Returns')

# Enhance plot aesthetics
plt.xlabel("Date", fontsize=14)
plt.ylabel("Daily Return", fontsize=14)
plt.title("Regression: Daily Returns vs. Daily Sentiment", fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.tight_layout()

# Add grid lines for clarity
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.show()



