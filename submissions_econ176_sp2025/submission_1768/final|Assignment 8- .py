# This will update Colab's libraries
#      and will avoid several DeprecationWarnings in the future...

import warnings
warnings.filterwarnings("ignore", category=ImportWarning, append=True)
warnings.filterwarnings("ignore", category=DeprecationWarning, append=True)

# these may need to be repeated, and the append parameter seems suspiciously unintuitive (i.e., not-working)
# this may not be needed
# # install --upgrade ipykernel








from web3 import Web3


# Check out whats available in the Web3 object, by
#       typing a period . after Web3 below
#       (make it Web3.  ~ then wait a moment...)

# Here, add a period (the selector operator) and wait a moment
#       a panel should popup to show all of the available fields

# Check out from_wei and to_wei !




from web3 import Web3

value_in_wei = Web3.to_wei(1, 'ether')
print(f"1 eth is {value_in_wei} wei.")


# converting to other units...
#            with the function Web3.from_wei

value_in_gwei = float(Web3.from_wei(42_000_000_000, 'gwei'))
value_in_gwei

# Python allows underscores instead of commas:
# gwei is short for "gigawei" which is 1 billion wei
# Note that, in Python, the underscore is a legal "thousands-separator"  (cool!)


#Challenge 1
import requests
from web3 import Web3

def ether_dollar_converter():
    try:
        # Fetch current ETH price from CoinGecko API
        url = 'https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd'
        response = requests.get(url)
        data = response.json()
        eth_price_usd = data['ethereum']['usd']

        # Calculations
        ether_in_dollars = eth_price_usd
        dollar_in_ether = 1 / eth_price_usd
        dollar_in_gwei = dollar_in_ether * Web3.to_wei(1, 'ether') / 1e9 # Convert to Gwei

        # Output
        print(f"1 ether is currently ${ether_in_dollars:.2f}")
        print(f"1 dollar is currently {dollar_in_ether:.15f} ether")
        print(f"1 dollar is about {dollar_in_gwei:.2f} gwei")

    except (requests.exceptions.RequestException, KeyError) as e:
        print(f"An error occurred: {e}")
        print("Please check your internet connection or the API endpoint.")


ether_dollar_converter()



<marquee style='width: 30%; color: Coral;'><b>It Works!</b></marquee>


#
# Here, we create a blockchain-connected Web3 object
#       Web3.EthereumTesterProvider() is our sandbox "provider"
#
# You WILL see some ImportWarnings...   (ignore those; they're Colab, not Ethereum or Web3)

provider = Web3.EthereumTesterProvider()
w3 = Web3(provider)
print("w3 is", w3)


w3.is_connected()


# Try using the completion panel and <tab> to find w3.eth.accounts ...




# These are the public addresses of the accounts we have

w3.eth.accounts


# how many accounts do we have?
len(w3.eth.accounts)


current_account = w3.eth.accounts[3]  # let's use #3

balance_in_wei = w3.eth.get_balance(current_account)

print(f"There are {balance_in_wei} wei in account #3.")


#
# Let's see it in floating point...

w = float(balance_in_wei)

print(f"There are a total of {w = } wei (in floating point), with 10**18 wei per ETH")


#
#  a small example of how to print all account balances
#          feel free to adapt or remove

#  this doesn't quite address the above challenge, but it's a start!

index = 0
for account in w3.eth.accounts:
    balance_in_wei = w3.eth.get_balance(account)
    balance_in_ETH = Web3.from_wei(balance_in_wei, 'ether')
    print(f"Account #{index}  amount: {balance_in_ETH}")
    index += 1



def total_ether_in_dollars():
    """
    Calculates the total value of Ether across all accounts in the Web3 provider,
    converts it to US dollars, prints the result, and returns the dollar value.
    """
    try:
        total_ether = 0
        for account in w3.eth.accounts:
            balance_wei = w3.eth.get_balance(account)
            balance_ether = Web3.from_wei(balance_wei, 'ether')
            total_ether += balance_ether

        # Fetch current ETH price (same as before)
        url = 'https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd'
        response = requests.get(url)
        data = response.json()
        eth_price_usd = data['ethereum']['usd']

        total_dollars = float(total_ether) * eth_price_usd
        print(f"Total Ether across all accounts: {total_ether:.6f} ETH")
        print(f"Total value in USD: ${total_dollars:.2f}")
        return total_dollars

    except (requests.exceptions.RequestException, KeyError, ValueError) as e:
        print(f"An error occurred: {e}")
        return 0  # Return 0 in case of error


total_ether_in_dollars()



# sometimes this helps...
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


#
# Let's see the latest block in our blockchain:

b = w3.eth.get_block('latest')   # this will be the "genesis" block

# Notice the number!


# Let's check the block's number, specifically
# Note that b is a dictionary:

print("Block b's number is ", b['number'])


# sometimes this helps with the warnings we don't want:
warnings.filterwarnings("ignore", category=ImportWarning)


#
# Let's create a single transaction:
#       Notice that the transaction is a dictionary
#       It's created, sent to the chain, and a hash is returned:

transaction = {                          # the transaction is a dictionary!
    'from': w3.eth.accounts[3],          # from acct 3
    'to': w3.eth.accounts[8],            # to acct 8
    'value': w3.to_wei(42000, 'ether')   # amount is 42,000 ether!
}

# now, we send the transaction to the blockchain:
tx_hash = w3.eth.send_transaction(transaction)

# let's look at its resulting hash
print("Done!\n")
print(f"The transaction hash is ... {tx_hash = }")


#
# getting the transaction is possible through that transaction hash

# It will return a dictionary of values:
d = w3.eth.get_transaction(tx_hash)



#
# let's get the block number from within those transaction details...

block_number = d['blockNumber']
print(f"{block_number = }")


#
# Then, let's get that block, from the blocknumber!

b = w3.eth.get_block(block_number)



#
# Which should be the same as getting the latest block:

b = w3.eth.get_block('latest')



#
#  a small example of how to print all account balances
#                  feel free to adapt or remove!

index = 0
for account in w3.eth.accounts:
    balance_in_wei = w3.eth.get_balance(account)
    balance_in_ETH = Web3.from_wei(balance_in_wei, 'ether')
    print(f"Account #{index}  amount: {balance_in_ETH} ether")
    index += 1


# create total, function, return, ...


#
# Let's create a single transaction for TOO MUCH ether
#       Notice that the transaction is a dictionary
#       It's created, sent to the chain, and a hash is returned:

transaction = {                          # the transaction is a dictionary!
    'from': w3.eth.accounts[3],          # from acct 3
    'to': w3.eth.accounts[8],            # to acct 8
    'value': w3.to_wei(42_000_000, 'ether')   # amount is 42 _million_ ether!
}

# now, we send the transaction to the blockchain:
tx_hash = w3.eth.send_transaction(transaction)

# let's look at its resulting hash
print("Done!\n")
tx_hash


#
# Let's create a single transaction for TOO MUCH ether
#       Notice that the transaction is a dictionary
#       It's created, sent to the chain, and a hash is returned:

from eth_utils import ValidationError

transaction = {                           # the transaction is a dictionary!
    'from': w3.eth.accounts[3],           # from acct 3
    'to': w3.eth.accounts[8],             # to acct 8
    'value': w3.to_wei(42_000, 'ether')   # change this to/from 42 _million_ ether!
}

# now, we send the transaction to the blockchain -- with a TRY/EXCEPT error-handler"
try:
    tx_hash = w3.eth.send_transaction(transaction)
    # let's look at its resulting hash
    print("Success!\n")
    print(f"The {tx_hash = }")
except ValidationError as e:
    print("Transaction failed: Insufficient funds.   This time we've _caught_ this exception. [[ Everything is fine... Nothing to see here... ]] \n")
    print(f"Here is the full ValidationError:\n    {e}")




import warnings
from web3 import Web3
import requests
from eth_utils import ValidationError
import random
import time
import matplotlib.pyplot as plt

def gini(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        for xj in x[i:]:
            diffsum += abs(xi - xj)
    return diffsum / (len(x)**2 * sum(x))


def run_transactions(num_transactions):
    insolvent_accounts = []
    GiniCoefficients = []
    AllBalances = []
    total_gas_used = 0
    for _ in range(num_transactions):
        balances = []
        for account in w3.eth.accounts:
            balances.append(w3.eth.get_balance(account))

        gini_coefficient = gini(balances)
        GiniCoefficients.append(gini_coefficient)
        AllBalances.append(balances.copy()) # Append a copy to avoid modification

        sender_account_index = random.choice([i for i in range(len(w3.eth.accounts)) if i not in insolvent_accounts])
        receiver_account_index = random.choice([i for i in range(len(w3.eth.accounts)) if i != sender_account_index and i not in insolvent_accounts])

        sender_account = w3.eth.accounts[sender_account_index]
        receiver_account = w3.eth.accounts[receiver_account_index]

        amount_ether = random.uniform(1, 500000)

        transaction = {
            'from': sender_account,
            'to': receiver_account,
            'value': w3.to_wei(amount_ether, 'ether')
        }
        try:
            tx_hash = w3.eth.send_transaction(transaction)
            print(f"Account #{sender_account_index} sent {amount_ether:.2f} ETH to Account #{receiver_account_index}")
            d = w3.eth.get_transaction(tx_hash)
            block_number = d['blockNumber']
            b = w3.eth.get_block(block_number)
            gas_used = b['gasUsed']
            total_gas_used += gas_used
            time.sleep(random.uniform(0.5, 1)) #reduced delay for faster execution.
        except ValidationError as e:
            print(f"Account #{sender_account_index} tried to send {amount_ether:.2f} ETH, but failed due to insufficient funds.")
            insolvent_accounts.append(sender_account_index)

    # Calculate final Gini coefficient
    final_balances = []
    for account in w3.eth.accounts:
      final_balances.append(w3.eth.get_balance(account))

    final_gini = gini(final_balances)
    GiniCoefficients.append(final_gini) #add final gini to the list
    AllBalances.append(final_balances)

    return GiniCoefficients, AllBalances, total_gas_used




#Tasks 1 and 2
provider = Web3.EthereumTesterProvider()
w3 = Web3(provider)
GiniCoefficients, AllBalances, total_gas_used = run_transactions(10) # Run 10 transactions
print("\nFinal Account Balances:")
for i, balance in enumerate(AllBalances[-1]):
    print(f"Account {i}: {Web3.from_wei(balance, 'ether')} ETH")

print(f"Final Gini Coefficient: {GiniCoefficients[-1]}")
print(f"Total Gas Used: {total_gas_used} wei")


#Task 3
provider = Web3.EthereumTesterProvider()
w3 = Web3(provider)
GiniCoefficients, AllBalances, total_gas_used = run_transactions(100) # Run 100 transactions
print("\nFinal Account Balances:")
for i, balance in enumerate(AllBalances[-1]):
    print(f"Account {i}: {Web3.from_wei(balance, 'ether')} ETH")

print(f"Final Gini Coefficient: {GiniCoefficients[-1]}")
print(f"Total Gas Used: {total_gas_used} wei")


#Task 4
import warnings
from web3 import Web3
import requests
from eth_utils import ValidationError
import random
import time
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

# Connect to the Ethereum tester provider
provider = Web3.EthereumTesterProvider()
w3 = Web3(provider)

def gini(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        for xj in x[i:]:
            diffsum += abs(xi - xj)
    return diffsum / (len(x)**2 * sum(x))

def run_tontine():
    insolvent_accounts = []
    GiniCoefficients = []
    AllBalances = []
    generations = 0

    while True:
        generations += 1
        balances = []
        for account in w3.eth.accounts:
            balances.append(w3.eth.get_balance(account))

        gini_coefficient = gini(balances)
        GiniCoefficients.append(gini_coefficient)
        AllBalances.append(balances.copy())

        positive_balance_count = sum(1 for balance in balances if balance > 0)
        if positive_balance_count <= 1:
            break

        valid_receivers = [i for i in range(len(w3.eth.accounts)) if i not in insolvent_accounts]
        if len(valid_receivers) <= 1:
            break

        sender_account_index = random.choice(valid_receivers)
        receiver_account_index = random.choice([i for i in valid_receivers if i != sender_account_index])

        sender_account = w3.eth.accounts[sender_account_index]
        receiver_account = w3.eth.accounts[receiver_account_index]

        amount_ether = random.uniform(1, 500000) # Adjust transaction amount range

        transaction = {
            'from': sender_account,
            'to': receiver_account,
            'value': w3.to_wei(amount_ether, 'ether')
        }

        try:
            tx_hash = w3.eth.send_transaction(transaction)
            print(f"Generation {generations}: Account #{sender_account_index} sent {amount_ether:.2f} ETH to Account #{receiver_account_index}")
            time.sleep(0.1)
        except ValidationError as e:
            print(f"Generation {generations}: Account #{sender_account_index} tried to send {amount_ether:.2f} ETH, but failed due to insufficient funds.")
            insolvent_accounts.append(sender_account_index)

    print(f"\nTontine completed in {generations} generations.")
    return GiniCoefficients, AllBalances, generations


# Run the Tontine simulation
GiniCoefficients, AllBalances, generations = run_tontine()


# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
for i in range(len(w3.eth.accounts)):
    plt.plot([Web3.from_wei(balance[i], 'ether') for balance in AllBalances])
plt.xlabel("Generations")
plt.ylabel("Ether Balance")
plt.title("Account Balances over Time")
plt.legend([f"Account {i}" for i in range(len(w3.eth.accounts))])

plt.subplot(1, 2, 2)
plt.plot(GiniCoefficients)
plt.xlabel("Generations")
plt.ylabel("Gini Coefficient")
plt.title("Gini Inequality over Time")


plt.tight_layout()
plt.show()



