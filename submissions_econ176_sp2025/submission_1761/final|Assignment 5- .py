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
  -H 'Authorization: Bearer EAAAl1v2YdTuCE6kMXfy5lVdklizjh9cm6PChPRk2H6AriWgexFexdaIQZvb3OWF' \
  -H 'Content-Type: application/json'


#
# Set an environment variable to YOUR access token

import os
os.environ['SQUARE_ACCESS_TOKEN'] = "EAAAl1v2YdTuCE6kMXfy5lVdklizjh9cm6PChPRk2H6AriWgexFexdaIQZvb3OWF"


# We can also use Colab's secrets. (This is not required. Just to show they're there.)

from google.colab import userdata
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
        print(f"{location['address']['loc[REDACTED]ty']}")
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
      "name": "Coffee Time, New New York ",
      "business_name": "Coffee Time",
      "address": {
        "address_line_1": "50 E 89th Street",
        "loc[REDACTED]ty": "New York",
        "administrative_district_level_1": "NY",
        "postal_code": "10128"
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

# [{'code': 'BAD_REQUEST', 'detail': 'Accept partial authorization not v[REDACTED]d for autocomplete payments.', 'category': 'INVALID_REQUEST_ERROR'}]


#
# first, a function that encapsulates _one_ random payment
#

import time, uuid, random
# we will assume there are global variables:
#    client
#    os.environ[']

EMAILS = [ "cff[REDACTED]", "cff@pitzer.edu", "cff@scripps.edu", "cff@pomona.edu", "cff[REDACTED]" ]
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

NUM_PAYMENTS = 5

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
  limit = 5
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



#Question 2

#using the same random choice of payments as above, we assume that one person named cff[REDACTED] made 5 payments instead
totalfeesOriginal = 0.0
for payment in result.body['payments']:
    amount = payment['total_money']['amount']
    fee_amt = payment['processing_fee'][0]['amount_money']['amount']   # we'll assume USD (actually it's cents!)
    fee_amt = float(fee_amt)
    #computed_fee = amount*0.029 + 30
    #print(f"{    fee_amt = }  ({amount = :4d})  ({computed_fee = :5.2f})")
    totalfeesOriginal += fee_amt

print(f"Total original fees: {totalfeesOriginal}")

#Now imagine that one person paid for everyone. take the total payment amount across the five transactions and apply Square's formula of 2.9% + 30c
#This allows cff to save on the 30 cent fixed fee per payment
totalNewFees = 0
totalAmount = 0
for payment in result.body['payments']:
    amount = payment['total_money']['amount']
    totalAmount += amount

totalNewFees = totalAmount * 0.029 + 30
print(f"Total new fees: {totalNewFees}")

people = len(result.body['payments'])

print("\nQuestion: How much would the Merchant gain if it created a CFF program where one person (the CFF) paid for all of their friends each time, resulting in fewer charges?")
print(f"Answer: The merchant gains {round(totalfeesOriginal - totalNewFees)} assuming the amount the customer pays stays constant. The CFF pays for {people} people")

print("\nQuestion: How much could be passed along to the CFF customer?")
print("Answer: Although the above assumes that the merchant takes 100% of the surplus, we considered splitting the surplus between the CFF and the merchant in order to incentivize the CFF to front the bill.")


#Question 3-6


#-------------------------------------------------------------------------------------------------------------------------
#creating a new store
result = client.locations.create_location(
  body = {
    "location": {
      "name": "Econ176_Participant_4's Katsudon Kingdom",
      "business_name": "Katsudon Kingdom",
      "address": {
        "address_line_1": "101 W 91st Street",
        "loc[REDACTED]ty": "New York",
        "administrative_district_level_1": "NY",
        "postal_code": "100024"
      }
    }
  }
)
result.status_code

#-------------------------------------------------------------------------------------------------------------------------
#creating new customers and orders (This menu is comprised of my favorite items at Katsuhama in NYC)
newEmails = [ "echen14[REDACTED]", "echen14@pitzer.edu", "echen14@scripps.edu", "echen14@pomona.edu", "echen14[REDACTED]", "emmettchen@yahoo.com", "emmettjchen@gmail.com" ]
newOrders = [ "Chicken Katsudon", "Pork Katsudon", "Beef Katsudon", "Gyoza", "Donburi", "Edamame", "Matcha" ]
newSizes = [ "Large", "Medium", "Kids" ]


#-------------------------------------------------------------------------------------------------------------------------
#creating a NEW payment structure using the random choices available in Katsudon Kingdom
def NEWrandom_payment(LOW=500, HIGH=900, real_request=False):
    """ one random payment to our business, CoffeeTime! """
    # create the details for a new payment:
    idempotency_key = str(uuid.uuid4())      # create an idempotency key

    amount = random.randint(LOW,HIGH)        # create the amount
    email = random.choice( newEmails )          # choose a customer
    order = random.choice( newSizes ) + " " + random.choice( newOrders )

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

#-------------------------------------------------------------------------------------------------------------------------
#making 5 orders
NUM_PAYMENTS = 5

for payment_number in range(NUM_PAYMENTS):
    print(f"Payment {payment_number}:")

    # make a random payment:
    result = NEWrandom_payment(real_request=True)

    # check result
    if type(result) == str:   # it's our string, "Not actually requested..."
       pass
    elif result.status_code != 200 or result.is_error():  # a real API error! Let's see it...
        print(f"{result.status_code=}")
        print(result.errors)



#Question 7-8

#-------------------------------------------------------------------------------------------------------------------------
#grabbing all payments (Q7)
result = client.payments.list_payments(
  limit = 5
)
result.status_code


#if result.is_success():
  #print(json.dumps(result.body, indent=2))
#elif result.is_error():
  #print(result.errors)


#-------------------------------------------------------------------------------------------------------------------------
#Analysis of the 5 payments (Q8)

min = 10000000
max = 0

for payment in result.body['payments']:

    amount = payment['total_money']['amount']
    if amount > max:
        max = amount
    if amount < min:
        min = amount

print(f"Analysis of my orders (completely prototyped and debugged):")
print(f"Most expensive order: {max}")
print(f"Cheapest order: {min}")




#Question 9

#-------------------------------------------------------------------------------------------------------------------------
#PAYMENT METHOD 1 (Higher fee paid by each individual)
totalfeesOriginal = 0.0
for payment in result.body['payments']:
    amount = payment['total_money']['amount']
    fee_amt = payment['processing_fee'][0]['amount_money']['amount']   # we'll assume USD (actually it's cents!)
    fee_amt = float(fee_amt)
    #computed_fee = amount*0.029 + 30
    #print(f"{    fee_amt = }  ({amount = :4d})  ({computed_fee = :5.2f})")
    totalfeesOriginal += fee_amt

print(f"Total original fees: {totalfeesOriginal}")

#-------------------------------------------------------------------------------------------------------------------------
#PAYMENT METHOD 2 (Lower fee on average, cumulative fee paid by one individual)
totalNewFees = 0
totalAmount = 0
for payment in result.body['payments']:
    amount = payment['total_money']['amount']
    totalAmount += amount

totalNewFees = totalAmount * 0.029 + 30
print(f"Total new fees: {round(totalNewFees)}")

print(f"Cost comparison: Assuming the customer pays the same amount, The merchant gains {round(totalfeesOriginal - totalNewFees)} in fees")
print("\nAgain, this assumes that the merchant takes 100% of the surplus but theoretically it could be split between the merchant and the CFF to incentivize the CFF to front the bill.")



#Question 10 (part 1)

#making 42 orders
NUM_PAYMENTS = 42

for payment_number in range(NUM_PAYMENTS):
    print(f"Payment {payment_number}:")

    # make a random payment:
    result = NEWrandom_payment(real_request=True)

    # check result
    if type(result) == str:   # it's our string, "Not actually requested..."
       pass
    elif result.status_code != 200 or result.is_error():  # a real API error! Let's see it...
        print(f"{result.status_code=}")
        print(result.errors)


#Question 10 (part 2)
result = client.payments.list_payments(
  limit = 42
)
result.status_code

min = 10000000
max = 0
for payment in result.body['payments']:

    amount = payment['total_money']['amount']
    if amount > max:
        max = amount
    if amount < min:
        min = amount

print(f"Analysis of my orders:")
print(f"Most expensive order: {max}")
print(f"Cheapest order: {min}")

totalfeesOriginal = 0.0
for payment in result.body['payments']:
    amount = payment['total_money']['amount']
    fee_amt = payment['processing_fee'][0]['amount_money']['amount']   # we'll assume USD (actually it's cents!)
    fee_amt = float(fee_amt)
    #computed_fee = amount*0.029 + 30
    #print(f"{    fee_amt = }  ({amount = :4d})  ({computed_fee = :5.2f})")
    totalfeesOriginal += fee_amt

print(f"Total original fees: {totalfeesOriginal}")

totalNewFees = 0
totalAmount = 0
for payment in result.body['payments']:
    amount = payment['total_money']['amount']
    totalAmount += amount

totalNewFees = totalAmount * 0.029 + 30
print(f"Total new fees: {round(totalNewFees)}")

people = len(result.body['payments'])

print(f"Cost comparison: Assuming the customer pays the same amount, The merchant gains {round(totalfeesOriginal - totalNewFees)} in fees. The CFF pays for {people} people")





#Question 11

print("We chose to compare the individual payment method to the cff method, where one person fronts the bill for the group purchases. This reduces the transaction fee paid to Square.")
print("Assuming we keep the customer payment amount the same, this reduction in transaction fee transfers a surplus of 'savings' which we plan to divide between the merchant and the CFF, creating a semi win-win.")
print("We assumed, for simplicity, that the CFF paid for all the transactions in the store. Because of this, we did not require additional information about how many people the purchaser was paying for; the CFF paid for all recorded transactions. Future work might entail IDing transactions as belonging to different groups each with their own CFF's. ")
print("Some variations could be made on how the 'surplus' money is split between the merchant and CFF. For example, in a scenario with many different groups, each with their own CFF, the surplus amount won't be that great. By offering a greater split and redsistributing the rest we could incentivize smaller groups to form into a larger group. This would increase the surplus, as we reduce one transaction fee per group paid to square. ")



