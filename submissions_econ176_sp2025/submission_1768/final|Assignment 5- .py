print("Our Business: Coffee Time!     (Our Consultancy: Pay Up...)")


#
# Reminder of how a "web-scraping" call works with requests
#

import requests

url = "https://www.cs.hmc.edu/~dodds/demo.html"   # this is sometimes called an "endpoint" ...
result = requests.get(url)

# if it succeeds, you should see <Response [200]>


#
# Notice that, as we explore, it's GOOD TO SEPARATE the request from its analysis...
#        We'll keep doing this...

print(result.text)   # the "demo page"


#
# curl is "easier" than requests and other programmatic methods
#
# The initial ! tells the notebook to run at the command-line, not Python:


# "getting" the output is trickier, however!


#
# We could use curl for API calls:


# Notice that the result is printed, not returned...


#
# curl is useful for its many parameters, e.g., -I gets the HTTP "headers"
#      which are unseen meta-data sent along with the visible part of a request:

# !curl -I  http://api.open-notify.org/iss-now.json




#
# Reminder of how API calls work using requests...

import requests

url = "http://api.open-notify.org/iss-now.json"   # this is sometimes called an "endpoint" ...
result = requests.get(url)

# if it succeeds, you should see <Response [200]>


#
# In this case, we know the result is a JSON file, and we can obtain it that way:

contents = result.json()      # needs to convert the text to a json dictionary...
print(f"The json data is {contents}")     # Aha!  Let's re/introduce f-strings...

# Take a look... what is a JSON object in Python terms?


#
# Let's re-remind ourselves how dictionaries work:

lat = contents['iss_position']['longitude']
lat = float(lat)
print(lat)


  -H 'Square-Version: 2025-02-20' \
  -H 'Authorization: Bearer <token_goes_here_get_rid_of_angle_brackets_too>' \
  -H 'Content-Type: application/json'


#
# Set an environment variable to YOUR access token

import os
os.environ['SQUARE_ACCESS_TOKEN'] = "EAAAl_gw7PnWlwXIsWRsOnF9lsXY9IGyntnIjBv2CWeIOd_M78OKJJILcVEws4H2"


# We can also use Colab's secrets. (This is not required. Just to show they're there.)

# from google.colab import userdata
# userdata.get('number')


# install the squareup library, if it's not already installed




#
# here is the SDK ...    (officially, Square's _Python_ SDK)

import square


from square.http.auth.o_auth_2 import BearerAuthCredentials
from square.client import Client
import os

client = Client(
    bearer_auth_credentials=BearerAuthCredentials(
        access_token=os.environ['SQUARE_ACCESS_TOKEN']    # You've likely included this above. Note: don't github it/release it!
    ),
    environment='sandbox')

# Old version uses only the access_token line.

# Let's see our client object:



# if we want to see all of the capabilities of our SDK:

for attribute in dir(client):
    if "__" not in attribute:   # skip the Python-specific ones ...
        print(attribute)        # let's see the Square-specific ones


#
# here is the API call to list the locations!

result = client.locations.list_locations()



# get attributes of the result object
print("Attributes (data and functions) of result:")
for item in dir(result):
    if "__" not in item:
        print(item)



# here's one that's data:
result.status_code


# here's one that's a function:
result.is_success()


# separating the network (API) call from the analysis of its results
#     means using result without re-retrieving it from Square. For example,

if result.is_success():
  print(f"result.body is", result.body)
elif result.is_error():
  print(f"result.errors are", result.errors)


# Let's print more readably.  To do so, let's use Python's JSON library:

import json

if result.is_success():
    print("result.body is")
    print(json.dumps(result.body, indent=2))
elif result.is_error():
    print("result.errors are")
    print(result.errors)


#
# further analysis of the "component pieces" of result:

if result.is_success():
    for location in result.body['locations']:
        print(f"{location['id']}: ", end="")
        print(f"{location['name']}, ", end="")
        print(f"{location['address']['address_line_1']}, ", end="")
        print(f"{location['address']['locality']}")
        if 'description' in location: print(f"{location['description']}")
        else: print("<no description>")

elif result.is_error():
    for error in result.errors:
        print(error['category'])
        print(error['code'])
        print(error['detail'])


#
# call!  (POST)  Please change this address for your own branch location!
#    Note: a _POST_ call requires a body of information that is sent in the API call:
#
# You should be wondering... how would I know what fields to include and how to include them?!
#    Answer:
# Square has an online API interface that helps with this!
#    It's here:   https://developer.squareup.com/explorer/square/locations-api/create-location

result = client.locations.create_location(
  body = {
    "location": {
      "name": "Coffee Time, Downtown Pittsburgh Branch ",
      "business_name": "Coffee Time",
      "description": "Stillers Swillers",
      "address": {
        "address_line_1": "326 Third Avenue",
        "locality": "Pittsburgh",
        "administrative_district_level_1": "PA",
        "postal_code": "15222"
      }
    }
  }
)
result.status_code



# separating the network (API) call from the analysis of its results:

import json

if result.is_success():
  print(json.dumps(result.body, indent=2))
elif result.is_error():
  print(result.errors)


#
# here is the API call to list the payments so far (with default parameters):

result = client.payments.list_payments()
result.status_code


# again, separating the network (API) call from the analysis of its results:

import json

if result.is_success():
  print(json.dumps(result.body, indent=2))
elif result.is_error():
  print(result.errors)


#
# create an idempotency key...
#
# in general, we'd keep this separate from the call below
# however, for a simulation of many payments, we _need_ a separate idempotency key for each!
#

# Python's library for unique identifiers is uuid:

import uuid
idempotency_key = str(uuid.uuid4())   # this generates a random value
print(f"idempotency_key: {idempotency_key}")


#
# With that in place, here's a bare-bones SDK call to create a payment:

result = client.payments.create_payment(
  body = {
    "source_id": "cnon:card-nonce-ok",
    "idempotency_key": idempotency_key,
    "amount_money": {
      "amount": 42,
      "currency": "USD"
    },
    "autocomplete": True,
    "accept_partial_authorization": False    # this and autocomplete can't both be True
  }
)
result.status_code


#
# Again, separating the network call (API) from the analysis of its result
#

import json

if result.is_success():
  print(json.dumps(result.body, indent=2))
elif result.is_error():
  print(result.errors)


#
# example of an error (copied over to this cell)
#

""" Here was the code from a create_payment call that had both "autocomplete" and "accept partial authorization"
        This combination of features is not permitted (as I found out!)


import json

if result.is_success():
  print(json.dumps(result.body, indent=2))
elif result.is_error():
  print(result.errors)
"""

# Here was the error message, printed from line 14 above:

# [{'code': 'BAD_REQUEST', 'detail': 'Accept partial authorization not valid for autocomplete payments.', 'category': 'INVALID_REQUEST_ERROR'}]


#
# first, a function that encapsulates _one_ random payment
#

import time, uuid, random
# we will assume there are global variables:
#    client
#    os.environ[']

EMAILS = [ "cff@cmc.edu", "cff@pitzer.edu", "cff@scripps.edu", "cff@pomona.edu", "cff@hmc.edu", "dodds@fun.com", "dass@cmc.com" ]
COFFEES = [ "Coffee", "PhilzPour", "Cometeer" ]
SIZES = [ "Venti", "Trenta", "Quarantadue" ]

def random_payment(LOW=500, HIGH=900, real_request=False):
    """ one random payment to our business, CoffeeTime! """
    # create the details for a new payment:
    idempotency_key = str(uuid.uuid4())      # create an idempotency key

    amount = random.randint(LOW,HIGH)        # create the amount
    email = random.choice( EMAILS )          # choose a customer
    order = random.choice( SIZES ) + " " + random.choice( COFFEES )

    # put the details into the dictionary we need:
    payment_details = {
        "source_id": "cnon:card-nonce-ok",
        "idempotency_key": idempotency_key,
        "amount_money": {
            "amount": amount,
            "currency": "USD"
        },
        "autocomplete": True,
        "note": order,
        "buyer_email_address": email,
    }

    # make the order, if requested
    result = "Not actually requested..."

    if real_request == True:
        # the client object needs to exist (from above!)
        result = client.payments.create_payment(body = payment_details) # here!
        time.sleep(random.randint(2,5))  # wait a few seconds

    print(f"  {email=} {order=} {amount=}")
    return result


#
# single-run testing of the above function, printing the result (if actually requested!)

result = random_payment(real_request=False)
print()
print(f"{result = }")
if result != "Not actually requested...":
    print(json.dumps(result.body, indent=2))


#
# a loop to make several payments  (by default, 5)

NUM_PAYMENTS = 100

for payment_number in range(NUM_PAYMENTS):
    print(f"Payment {payment_number}:")

    # make a random payment:
    result = random_payment(real_request=False)

    # check result
    if type(result) == str:   # it's our string, "Not actually requested..."
       pass
    elif result.status_code != 200 or result.is_error():  # a real API error! Let's see it...
        print(f"{result.status_code=}")
        print(result.errors)



#
# here, we _get_ the last five payments:
 #      (another API call, naturally)
 #      (via SDK, equally naturally!)

result = client.payments.list_payments(
  limit = 100
)
result.status_code


#
# Let's see the result...

if result.is_success():
  print(json.dumps(result.body, indent=2))
elif result.is_error():
  print(result.errors)


#
# Yikes! There _seem_ to be a lot of keys...
print(list(result.body.keys()))

# Aha, actually not so many...


# opportunity to check for all of the keys/fields in each payment:

for payment in result.body['payments']:
    for key in payment.keys():
        print(key)
    break


#
# loop over them and get the amount and email for each

for payment in result.body['payments']:

    amount = payment['total_money']['amount']   # to hold the amount of this transaction

    # an email variable to hold the email -- or a default string, if there is no email...
    if 'buyer_email_address' in payment:  email = payment['buyer_email_address']
    else:                                 email = "<not provided>"

    print("Amount:", amount)
    print(" Email:   ", email)


#
# My analysis is to count the per-customer average of CMC coffee costs:

total_buyers = 0
num_cmcers = 0
total_cmc_payments = 0
total_payments = 0

for payment in result.body['payments']:
    total_buyers += 1

    amount = payment['total_money']['amount']
    print("Amount:", amount)

    if 'buyer_email_address' in payment:  email = payment['buyer_email_address']
    else:                                 email = "<not provided>"
    print(" Email:   ", email)

    if "cmc" in email:   # is the email from cmc?
        num_cmcers += 1
        total_cmc_payments += amount  # the amount
    total_payments += amount  # the amount

print()
print(f"Total # of CMCers:  {num_cmcers = }")
print(f"Total CMC coffees:  {total_cmc_payments = }")
print(f"Total coffees:      {total_payments = }")

print()

if num_cmcers > 0:  # were there any CMCers?
    cmc_av = total_cmc_payments/num_cmcers
    print(f"Average for CMC:    {cmc_av}")
    full_av = total_payments/total_buyers
    print(f"Average overall:    {full_av}")
else:
    print("Failure: There were no CMC'ers!  Sell more coffee!!")




# Let's see square's processing fees...

for payment in result.body['payments']:
  print(payment['processing_fee'])


# Let's total up square's processing fees: the API is at 2.9 + 30¢

total = 0.0
for payment in result.body['payments']:
    amount = payment['total_money']['amount']
    fee_amt = payment['processing_fee'][0]['amount_money']['amount']   # we'll assume USD (actually it's cents!)
    fee_amt = float(fee_amt)
    computed_fee = amount*0.029 + 30
    print(f"{    fee_amt = }  ({amount = :4d})  ({computed_fee = :5.2f})")
    total += fee_amt

print(f"\nTotal fees:\n     {total = }")


#CoffeeCashProgram
import collections
from datetime import datetime
total_fees_with_square = collections.defaultdict(lambda: collections.defaultdict(float))
week_fees = collections.defaultdict(lambda: collections.defaultdict(float))
for payment in result.body['payments']:
   date = datetime.strptime(payment["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ")
   week_number = date.isocalendar()[1]
   if 'buyer_email_address' in payment:  email = payment['buyer_email_address']
   else:                                 email = "<not provided>"

   if email in week_fees:
      week_fees[email][week_number] += payment['total_money']['amount']
   else:
      week_fees[email][week_number] = payment['total_money']['amount']

   if email in total_fees_with_square:
      total_fees_with_square[email][week_number] += payment['processing_fee'][0]['amount_money']['amount']
   else:
      total_fees_with_square[email][week_number] = payment['processing_fee'][0]['amount_money']['amount']

print("Total amount of money spent by email: ")
print({k: dict(v) for k, v in week_fees.items()})
print("Total amount of fees accumulated by email:")
print({k: dict(v) for k, v in total_fees_with_square.items()})


for email in week_fees:
  for week in week_fees[email]:
      week_fees[email][week] = week_fees[email][week] *0.029 + 30

print("Total amount of fee if paid weekly by email:")
print({k: dict(v) for k, v in week_fees.items()})


print("\n")

for email in total_fees_with_square:
  for week in total_fees_with_square[email]:
    print("Merchant" + " would have saved $" + str(total_fees_with_square[email][week] - week_fees[email][week]) + " if " + email + " used CoffeeCash and the transaction was ran weekly.")

print("\n")

sum_week = 0
sum_fee_week = 0
for email in week_fees:
  for week in week_fees[email]:
    sum_week += week_fees[email][week]
    sum_fee_week += total_fees_with_square[email][week]
    print(email + "'s payments created $" + str(week_fees[email][week]) + " in fees in week " + str(week))

print("\n")
print("Total saved by merchant with weekly payments")
print("$" + str(sum_fee_week - sum_week))





import collections
from datetime import datetime
import matplotlib.pyplot as plt

# Now, let's plot the data
# We'll keep track of how the fees accumulate as the number of customers increases over time
customer_count = 0
cumulative_weekly_fees = []
cumulative_payment_fees = []

# Iterate over the week_fees and total_fees_with_square to calculate cumulative data
for email in week_fees:
    for week in sorted(week_fees[email].keys()):
        customer_count += 1

        cumulative_weekly_fees.append((customer_count, week_fees[email][week]))

        cumulative_payment_fees.append((customer_count, total_fees_with_square[email][week]))


plt.figure(figsize=(10, 6))


week_customers, cumulative_weekly = zip(*cumulative_weekly_fees)
plt.plot(week_customers, cumulative_weekly, label="Weekly Fees", marker='o', color='blue')


payment_customers, cumulative_payment = zip(*cumulative_payment_fees)
plt.plot(payment_customers, cumulative_payment, label="Payment Fees", marker='x', color='red')


plt.xlabel('Number of Customers')
plt.ylabel('Amount in Fees ($)')
plt.title('Cumulative Fees: Weekly vs Payment Processing Fees')
plt.legend()
plt.grid(True)


plt.show()


