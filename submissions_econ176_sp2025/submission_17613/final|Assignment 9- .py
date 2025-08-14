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

Web3.from_wei


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


#
# Feel free to create your own cells for this challenge...
#


import requests
import json

def get_eth_price():
    try:
        url = 'https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd'
        response = requests.get(url)
        response.raise_for_status()

        data = json.loads(response.text)
        eth_price = data['ethereum']['usd']
        return eth_price
    except requests.exceptions.RequestException as e:
        print(f"Error fetching ETH price: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error processing ETH price data: {e}")
        return None

def currency_converter():
    eth_price = get_eth_price()
    if eth_price is not None:
        print(f"1 ether is currently ${eth_price:.2f}")

        dollar_to_eth = 1 / eth_price
        print(f"1 dollar is currently {dollar_to_eth} ether")

        dollar_to_gwei = dollar_to_eth * (10**9)
        print(f"1 dollar is about {dollar_to_gwei:.2f} gwei")

currency_converter()



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


from decimal import Decimal

def calculate_total_eth_in_dollars(w3, eth_to_usd_rate):
    total_eth = Decimal('0')
    for account in w3.eth.accounts:
        balance_in_wei = w3.eth.get_balance(account)
        balance_in_eth = Web3.from_wei(balance_in_wei, 'ether')
        total_eth += balance_in_eth

    eth_to_usd_rate = Decimal(str(eth_to_usd_rate))
    total_dollars = total_eth * eth_to_usd_rate

    print(f"Total across all accounts: ${total_dollars:,.2f} USD")
    return total_dollars

current_eth_price = get_eth_price()
total_usd = calculate_total_eth_in_dollars(w3, current_eth_price)


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




import random
import time
import numpy as np
import matplotlib.pyplot as plt
from web3 import Web3

provider = Web3.EthereumTesterProvider()
w3 = Web3(provider)

def gini_coefficient(balances):
    balances = np.array(balances)
    n = len(balances)
    if n == 0:
        return 0
    diff_sum = 0
    for i in range(n):
        for j in range(n):
            diff_sum += abs(balances[i] - balances[j])
    return diff_sum / (2 * n * np.sum(balances))


def run_marketplace(num_transactions, delay_seconds=None):
    solvent_accounts = w3.eth.accounts.copy()
    gas_used_total = 0
    gini_history = []

    for i in range(num_transactions):
        if len(solvent_accounts) < 2:
            break

        sender, receiver = random.sample(solvent_accounts, 2)

        sender_balance = w3.eth.get_balance(sender)
        amount = random.randint(1, int(0.1 * sender_balance))

        print(f"Transaction {i+1}: Account {w3.eth.accounts.index(sender)} sending {Web3.from_wei(amount, 'ether'):.2f} ETH to Account {w3.eth.accounts.index(receiver)}")

        try:
            tx_hash = w3.eth.send_transaction({
                'from': sender,
                'to': receiver,
                'value': amount,
                'gas': 21000
            })

            receipt = w3.eth.get_transaction_receipt(tx_hash)
            gas_used_total += receipt['gasUsed']

        except Exception as e:
            print(f"Transaction failed: {e}")
            solvent_accounts.remove(sender)
            continue

        if delay_seconds is not None:
            time.sleep(delay_seconds)

        if i % 10 == 0:
            balances = [w3.eth.get_balance(acc) for acc in w3.eth.accounts]
            gini = gini_coefficient(balances)
            gini_history.append(gini)

    print("\nFinal Balances:")
    total = 0
    balances = []
    for i, account in enumerate(w3.eth.accounts):
        balance = w3.eth.get_balance(account)
        eth_balance = Web3.from_wei(balance, 'ether')
        balances.append(eth_balance)
        total += eth_balance
        print(f"Account {i}: {eth_balance:.2f} ETH")

    print(f"\nTotal ETH: {total:.2f}")
    print(f"Total Gas Used: {gas_used_total}")
    print(f"Final Gini Coefficient: {gini_coefficient(balances):.4f}")

    return gini_history

gini_history_10 = run_marketplace(10, delay_seconds=2)

gini_history_100 = run_marketplace(100, delay_seconds=0.001)

plt.figure(figsize=(10, 5))
plt.plot(gini_history_100)
plt.title("Gini Coefficient Over 100 Transactions")
plt.xlabel("Transaction Batch (every 10 tx)")
plt.ylabel("Gini Coefficient")
plt.grid()
plt.show()


import matplotlib.pyplot as plt
import numpy as np
import random
from web3 import Web3

def gini_coefficient(balances):
    balances = np.array(balances)
    n = len(balances)
    if n == 0 or np.sum(balances) == 0:
        return 0
    diff_sum = 0
    for i in range(n):
        for j in range(n):
            diff_sum += abs(balances[i] - balances[j])
    return diff_sum / (2 * n * np.sum(balances))

def run_tontine_with_metrics():
    accounts = w3.eth.accounts.copy()
    active_accounts = accounts.copy()
    generation = 0

    all_generations = []
    all_balances = []
    all_gini = []

    initial_balances = [w3.eth.get_balance(acc) for acc in accounts]
    all_balances.append(initial_balances)
    all_gini.append(gini_coefficient(initial_balances))

    while len([acc for acc in active_accounts if w3.eth.get_balance(acc) > 0]) > 1:
        generation += 1
        print(f"\nGeneration {generation} - {len(active_accounts)} active accounts")

        for _ in range(min(5, len(active_accounts))):  # Fewer transactions per gen
            if len(active_accounts) < 2:
                break

            balances = {acc: w3.eth.get_balance(acc) for acc in active_accounts}

            send_weights = [1/(balances[acc]+1) for acc in active_accounts]
            sender = random.choices(active_accounts, weights=send_weights, k=1)[0]

            if random.random() > 0.2:
                receiver = max(active_accounts, key=lambda acc: balances[acc])
            else:
                receiver = random.choice([acc for acc in active_accounts if acc != sender])

            current_gini = gini_coefficient(list(balances.values()))
            sender_balance = balances[sender]

            if current_gini > 0.7:
                amount = random.randint(int(0.8 * sender_balance), sender_balance)
            else:
                amount = random.randint(int(0.3 * sender_balance), int(0.7 * sender_balance))

            try:
                tx_hash = w3.eth.send_transaction({
                    'from': sender,
                    'to': receiver,
                    'value': amount,
                    'gas': 21000
                })

                if w3.eth.get_balance(sender) == 0:
                    active_accounts.remove(sender)

            except Exception as e:
                if sender in active_accounts:
                    active_accounts.remove(sender)
                continue

        current_balances = [w3.eth.get_balance(acc) for acc in accounts]
        all_balances.append(current_balances)
        current_gini = gini_coefficient(current_balances)
        all_gini.append(current_gini)
        all_generations.append(generation)

        print(f"Gini coefficient: {current_gini:.4f}")
        print("Top 3 accounts:")
        sorted_accounts = sorted([(i, bal) for i, bal in enumerate(current_balances)],
                                key=lambda x: x[1], reverse=True)[:3]
        for idx, bal in sorted_accounts:
            print(f"  Account {idx}: {Web3.from_wei(bal, 'ether'):.2f} ETH")

    print(f"\n🏆 Tontine completed in {generation} generations!")

    eth_balances = np.array([[Web3.from_wei(b, 'ether') for b in gen] for gen in all_balances])

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    for i in range(len(accounts)):
        plt.plot(range(len(all_balances)), eth_balances[:, i], label=f"Account {i}")
    plt.title("Account Balances Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("ETH Balance")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(len(all_gini)), all_gini, 'r-', linewidth=2)
    plt.title("Wealth Inequality (Gini Coefficient)")
    plt.xlabel("Generation")
    plt.ylabel("Gini Coefficient")
    plt.ylim(0, 1)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return generation, all_balances, all_gini

generations, balances, gini_values = run_tontine_with_metrics()


# Example of resetting the universe and running the results:
provider = Web3.EthereumTesterProvider()
w3 = Web3(provider)

GiniCoefficients, AllBalances = run_transactions()


