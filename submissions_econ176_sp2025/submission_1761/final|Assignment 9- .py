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

# Web3.



# from web3 import Web3

value_in_wei = Web3.to_wei(1, 'ether')
print(f"1 eth is {value_in_wei} wei.")


# converting to other units...
#            with the function Web3.from_wei

value_in_gwei = float(Web3.from_wei(42_000_000_000, 'gwei'))
value_in_gwei

# Python allows underscores instead of commas:
# gwei is short for "gigawei" which is 1 billion wei
# Note that, in Python, the underscore is a legal "thousands-separator"  (cool!)


import requests

def currency_value():
  ether = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd')
  dollar_ether = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=eth')



import requests
from web3 import Web3

def currency_value():
    try:
        ether_price = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd').json()
        eth_usd_price = ether_price['ethereum']['usd']

        # Calculate ether value in dollars
        ether_in_dollars = eth_usd_price

        # Calculate dollar value in ether
        dollar_in_ether = 1 / eth_usd_price

        # Calculate dollar value in gwei
        dollar_in_gwei = dollar_in_ether * (10**9)

        print(f"1 ether is currently ${ether_in_dollars:.2f}")
        print(f"1 dollar is currently {dollar_in_ether} ether")
        print(f"1 dollar is about {dollar_in_gwei:.2f} gwei")

    except (requests.exceptions.RequestException, KeyError) as e:
        print(f"An error occurred: {e}")

currency_value()



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


import requests
from web3 import Web3

def eth_manager():
    """
    Finds the amount of Ether in all accounts, adds it up,
    converts it to dollars, prints the total, and returns that value.
    """
    try:
        # Get current ETH price in USD
        response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd')
        eth_price_usd = response.json()['ethereum']['usd']

        # Calculate total ETH across all accounts
        total_eth = 0
        for i, account in enumerate(w3.eth.accounts):
            balance_in_wei = w3.eth.get_balance(account)
            balance_in_eth = Web3.from_wei(balance_in_wei, 'ether')
            total_eth += balance_in_eth
            print(f"Account #{i}: {balance_in_eth} ETH")

        # Convert to dollars
        total_dollars = total_eth * eth_price_usd

        print(f"\nTotal ETH: {total_eth}")
        print(f"Current ETH price: ${eth_price_usd}")
        print(f"Total value in USD: ${total_dollars:,.2f}")

        return total_dollars

    except Exception as e:
        print(f"Error occurred: {e}")
        return 0

# Test the function
total_value = eth_manager()


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




import requests
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from web3 import Web3
from eth_utils import ValidationError
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize Web3 with the tester provider (a simulated blockchain)
provider = Web3.EthereumTesterProvider()
w3 = Web3(provider)



def run_simple_marketplace(num_transactions=10, delay_min=2, delay_max=3,
                         min_amount=1, max_amount=100):
    """
    Task 1: Run N transactions with random accounts and amounts

    Parameters:
    - num_transactions: Number of transactions to perform
    - delay_min/delay_max: Min/max seconds between transactions
    - min_amount/max_amount: Min/max transaction amount (in ether)
    """
    print(f"\n=== Running {num_transactions} random transactions ===\n")

    for i in range(num_transactions):
        # Choose random sender and recipient
        sender_idx = random.randint(0, len(w3.eth.accounts) - 1)
        recipient_idx = random.randint(0, len(w3.eth.accounts) - 1)

        # Make sure sender and recipient are different
        while recipient_idx == sender_idx:
            recipient_idx = random.randint(0, len(w3.eth.accounts) - 1)

        sender = w3.eth.accounts[sender_idx]
        recipient = w3.eth.accounts[recipient_idx]

        # Generate random amount (in ether, then convert to wei)
        amount_eth = random.uniform(min_amount, max_amount)
        amount_wei = w3.to_wei(amount_eth, 'ether')

        # Build transaction
        transaction = {
            'from': sender,
            'to': recipient,
            'value': amount_wei
        }

        # Print what's happening
        print(f"Transaction {i+1}: Account {sender_idx} is sending {amount_eth:.4f} ether to Account {recipient_idx}")

        try:
            # Send transaction
            tx_hash = w3.eth.send_transaction(transaction)
            print(f"  Transaction successful: {tx_hash.hex()[:10]}...")

        except ValidationError as e:
            print(f"  Transaction failed: {e}")

        # Wait between transactions
        delay = random.uniform(delay_min, delay_max)
        time.sleep(delay)

    print("\nAll transactions completed!")


def run_small_marketplace(num_transactions=10):
    """
    Task 2: Run a small marketplace and track stats like gas and inequality
    """
    print(f"\n=== Running small marketplace ({num_transactions} transactions) ===\n")

    # Track total gas used
    total_gas_used = 0

    # Store initial balances
    initial_balances = [w3.eth.get_balance(account) for account in w3.eth.accounts]
    initial_gini = calculate_gini(initial_balances)
    initial_variance = calculate_variance(initial_balances)

    print(f"Initial Gini coefficient: {initial_gini:.4f}")
    print(f"Initial variance: {initial_variance:.2f} ETH²\n")

    # Run transactions
    for i in range(num_transactions):
        # Choose random sender and recipient
        sender_idx = random.randint(0, len(w3.eth.accounts) - 1)
        recipient_idx = random.randint(0, len(w3.eth.accounts) - 1)

        # Make sure sender and recipient are different
        while recipient_idx == sender_idx:
            recipient_idx = random.randint(0, len(w3.eth.accounts) - 1)

        sender = w3.eth.accounts[sender_idx]
        recipient = w3.eth.accounts[recipient_idx]

        # Generate random amount (in ether, then convert to wei)
        amount_eth = random.uniform(1, 100)
        amount_wei = w3.to_wei(amount_eth, 'ether')

        # Build transaction
        transaction = {
            'from': sender,
            'to': recipient,
            'value': amount_wei
        }

        # Print what's happening
        print(f"Transaction {i+1}: Account {sender_idx} is sending {amount_eth:.4f} ether to Account {recipient_idx}")

        try:
            # Send transaction
            tx_hash = w3.eth.send_transaction(transaction)

            # Get transaction receipt for gas information
            tx_receipt = w3.eth.get_transaction_receipt(tx_hash)
            gas_used = tx_receipt.gasUsed
            total_gas_used += gas_used

            print(f"  Transaction successful - Gas used: {gas_used}")

        except ValidationError as e:
            print(f"  Transaction failed: {e}")

        # Short delay for readability
        time.sleep(0.1)

    # Calculate final stats
    final_balances = [w3.eth.get_balance(account) for account in w3.eth.accounts]
    final_gini = calculate_gini(final_balances)
    final_variance = calculate_variance(final_balances)

    # Print final account balances
    print("\nFinal Account Balances:")
    total_eth = 0
    for idx, account in enumerate(w3.eth.accounts):
        balance_in_wei = w3.eth.get_balance(account)
        balance_in_eth = Web3.from_wei(balance_in_wei, 'ether')
        total_eth += balance_in_eth
        print(f"Account #{idx}: {balance_in_eth:.4f} ETH")

    print(f"\nTotal ETH across all accounts: {total_eth:.4f}")
    print(f"Total gas used: {total_gas_used}")
    print(f"Final Gini coefficient: {final_gini:.4f}")
    print(f"Final variance: {final_variance:.2f} ETH²")

    return total_gas_used, final_gini, final_variance



def run_large_marketplace(num_transactions=100):
    """
    Task 3: Run a large marketplace with insolvency handling

    When an account tries to overspend, it becomes insolvent and
    is removed from future transactions.
    """
    print(f"\n=== Running large marketplace ({num_transactions} transactions) ===\n")

    # Track total gas used
    total_gas_used = 0

    # Track solvent accounts (all are initially solvent)
    solvent_accounts = list(range(len(w3.eth.accounts)))

    # Store initial balances
    initial_balances = [w3.eth.get_balance(account) for account in w3.eth.accounts]
    initial_gini = calculate_gini(initial_balances)

    print(f"Initial number of solvent accounts: {len(solvent_accounts)}")
    print(f"Initial Gini coefficient: {initial_gini:.4f}\n")

    # Run transactions
    for i in range(num_transactions):
        if len(solvent_accounts) <= 1:
            print("Only one solvent account remains! Stopping transactions.")
            break

        # Choose random sender and recipient from solvent accounts
        sender_idx = random.choice(solvent_accounts)
        recipient_options = [idx for idx in solvent_accounts if idx != sender_idx]
        recipient_idx = random.choice(recipient_options)

        sender = w3.eth.accounts[sender_idx]
        recipient = w3.eth.accounts[recipient_idx]

        # Get sender's balance
        sender_balance = w3.eth.get_balance(sender)

        # Make the transaction amount significantly high to increase chances of insolvency
        # 50% to 90% of available balance
        percentage = random.uniform(0.5, 0.9)
        amount = int(sender_balance * percentage)

        # Build transaction
        transaction = {
            'from': sender,
            'to': recipient,
            'value': amount
        }

        # Print what's happening
        print(f"Transaction {i+1}: Account {sender_idx} is sending {Web3.from_wei(amount, 'ether'):.4f} ether to Account {recipient_idx}")

        try:
            # Send transaction
            tx_hash = w3.eth.send_transaction(transaction)

            # Get transaction receipt for gas information
            tx_receipt = w3.eth.get_transaction_receipt(tx_hash)
            gas_used = tx_receipt.gasUsed
            total_gas_used += gas_used

            print(f"  Transaction successful - Gas used: {gas_used}")

        except ValidationError as e:
            print(f"  Transaction failed: {e}")

            # If insufficient funds, mark account as insolvent
            if "insufficient funds" in str(e):
                print(f"  Account {sender_idx} is now insolvent and will be removed from future transactions")
                solvent_accounts.remove(sender_idx)

        # Short delay for readability
        time.sleep(0.01)

        # Print status update every 10 transactions
        if (i+1) % 10 == 0:
            print(f"\nStatus after {i+1} transactions:")
            print(f"  Solvent accounts remaining: {len(solvent_accounts)}")
            current_balances = [w3.eth.get_balance(account) for account in w3.eth.accounts]
            current_gini = calculate_gini(current_balances)
            print(f"  Current Gini coefficient: {current_gini:.4f}\n")

    # Calculate final stats
    final_balances = [w3.eth.get_balance(account) for account in w3.eth.accounts]
    final_gini = calculate_gini(final_balances)

    # Print final account balances
    print("\nFinal Account Balances:")
    total_eth = 0
    for idx, account in enumerate(w3.eth.accounts):
        balance_in_wei = w3.eth.get_balance(account)
        balance_in_eth = Web3.from_wei(balance_in_wei, 'ether')
        total_eth += balance_in_eth
        print(f"Account #{idx}: {balance_in_eth:.4f} ETH")

    print(f"\nTotal ETH across all accounts: {total_eth:.4f}")
    print(f"Total gas used: {total_gas_used}")
    print(f"Final number of solvent accounts: {len(solvent_accounts)}")
    print(f"Final Gini coefficient: {final_gini:.4f}")

    return total_gas_used, final_gini, solvent_accounts


def run_tontine():
    """
    Finale: Run a tontine marketplace until only one account has a positive balance.

    Track and plot:
    - How many generations it takes
    - Inequality measure (Gini coefficient) over time
    - Account balances over time
    """
    print("\n=== Running Tontine Simulation ===\n")

    # Reset the blockchain for clean initial state
    provider = Web3.EthereumTesterProvider()
    w3 = Web3(provider)

    # Set all accounts to equal initial balance for fair start
    initial_balance = w3.to_wei(1000, 'ether')  # 1000 ETH each
    for account in w3.eth.accounts[1:]:  # Use first account as source
        w3.eth.send_transaction({
            'from': w3.eth.accounts[0],
            'to': account,
            'value': initial_balance
        })

    # Initialize tracking variables
    GiniCoefficients = []
    AllBalances = []
    generations = 0
    solvent_accounts = list(range(len(w3.eth.accounts)))

    # Get initial balances
    initial_balances = [w3.eth.get_balance(account) for account in w3.eth.accounts]
    AllBalances.append(initial_balances)

    # Calculate initial Gini coefficient
    initial_gini = calculate_gini(initial_balances)
    GiniCoefficients.append(initial_gini)

    print("Starting tontine simulation...")
    print("Initial number of solvent accounts:", len(solvent_accounts))
    print(f"Initial Gini coefficient: {initial_gini:.4f}\n")

    # Run until only one solvent account remains
    while len(solvent_accounts) > 1:
        generations += 1

        # Choose random sender and recipient from solvent accounts
        sender_idx = random.choice(solvent_accounts)
        recipient_options = [idx for idx in solvent_accounts if idx != sender_idx]
        recipient_idx = random.choice(recipient_options)

        sender = w3.eth.accounts[sender_idx]
        recipient = w3.eth.accounts[recipient_idx]

        # Get sender's balance
        sender_balance = w3.eth.get_balance(sender)

        # Set transaction amount to 40-80% of balance
        # This makes the tontine progress more quickly
        amount = int(sender_balance * random.uniform(0.4, 0.8))

        # Build transaction
        transaction = {
            'from': sender,
            'to': recipient,
            'value': amount
        }

        try:
            # Send transaction
            tx_hash = w3.eth.send_transaction(transaction)

            if generations % 10 == 0 or generations < 10:
                print(f"Generation {generations}: Account {sender_idx} sent {Web3.from_wei(amount, 'ether'):.2f} ETH to Account {recipient_idx}")

        except ValidationError as e:
            # If insufficient funds, mark account as insolvent
            if "insufficient funds" in str(e):
                if generations % 10 == 0 or generations < 10:
                    print(f"Generation {generations}: Account {sender_idx} is now insolvent")
                solvent_accounts.remove(sender_idx)

        # Get updated balances
        current_balances = [w3.eth.get_balance(account) for account in w3.eth.accounts]
        AllBalances.append(current_balances)

        # Calculate Gini coefficient
        gini = calculate_gini(current_balances)
        GiniCoefficients.append(gini)

        # Print status update every 10 generations
        if generations % 20 == 0:
            print(f"\nGeneration {generations} status:")
            print(f"  Solvent accounts remaining: {len(solvent_accounts)}")
            print(f"  Current Gini coefficient: {gini:.4f}")

    # Print winner
    winner_idx = solvent_accounts[0]
    winner_balance = Web3.from_wei(w3.eth.get_balance(w3.eth.accounts[winner_idx]), 'ether')

    print(f"\nTontine complete after {generations} generations!")
    print(f"Account #{winner_idx} wins with {winner_balance:.2f} ETH")
    print(f"Final Gini coefficient: {GiniCoefficients[-1]:.4f}")

    # Plot the results
    plot_tontine_results(GiniCoefficients, AllBalances)

    while len(solvent_accounts) > 1 and generations < max_generations and gini < 0.95: # Add a Gini threshold <--- Modify here
        generations += 1

    return generations, GiniCoefficients, AllBalances

def plot_tontine_results(GiniCoefficients, AllBalances):
    """Plot the results of the tontine simulation."""
    # Create figure with two subplots
    plt.figure(figsize=(12, 10))

    # Plot 1: Account balances over time
    plt.subplot(2, 1, 1)
    generations = range(len(AllBalances))
    balances_in_eth = [[float(Web3.from_wei(balance, 'ether')) for balance in gen_balances]
                      for gen_balances in AllBalances]

    for i in range(len(w3.eth.accounts)):
        account_balances = [balances[i] for balances in balances_in_eth]
        plt.plot(generations, account_balances, label=f"Account {i}")

    plt.title("Account Balances Over Time")
    plt.xlabel("Generation")
    plt.ylabel("Balance (ETH)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)

    # Plot 2: Gini coefficient over time
    plt.subplot(2, 1, 2)
    plt.plot(generations, GiniCoefficients, 'r-', linewidth=2)
    plt.title("Gini Coefficient Over Time")
    plt.xlabel("Generation")
    plt.ylabel("Gini Coefficient")
    plt.grid(True)

    plt.tight_layout()
    plt.show()




def run_challenge3():
    print("====== CHALLENGE #3: COUNT YOUR GOLD ======")
    total_value = count_my_gold()
    print("\n")
    return total_value

def run_challenge4():
    # Reset for clean state
    provider = Web3.EthereumTesterProvider()
    w3 = Web3(provider)

    print("====== CHALLENGE #4: A BUSTLING MARKETPLACE ======")

    # Task 1: Simple transactions
    print("\n--- Task 1: Simple Transactions ---")
    run_simple_marketplace(num_transactions=5, delay_min=0.5, delay_max=1)

    # Task 2: Small marketplace with stats
    print("\n--- Task 2: Small Marketplace with Stats ---")
    gas_used, gini, variance = run_small_marketplace(num_transactions=10)

    # Task 3: Large marketplace with insolvency
    print("\n--- Task 3: Large Marketplace with Insolvency ---")
    gas_used, gini, solvent = run_large_marketplace(num_transactions=50)

    # Finale: Tontine
    print("\n--- Finale: The Tontine ---")
    generations, gini_history, balance_history = run_tontine()

    # Overall reflection
    print("\n====== REFLECTION ======")
    print("This marketplace simulation demonstrates several key blockchain concepts:")
    print("1. How transactions redistribute wealth among accounts")
    print("2. How the Gini coefficient measures inequality in the system")
    print("3. The natural tendency for wealth to concentrate when random transactions occur")
    print("4. How a tontine ultimately results in a single winner through generations of transactions")



import numpy as np

def calculate_gini(balances):
    """Calculate the Gini coefficient of a list of account balances."""
    balances = np.array(balances)
    balances = balances[balances != 0]  # Remove zero balances to avoid division by zero
    if len(balances) == 0:
        return 0  # Handle the case where all balances are zero

    # Sort balances in ascending order
    sorted_balances = np.sort(balances)

    # Calculate the cumulative sum of balances
    cumulative_balances = np.cumsum(sorted_balances)

    # Calculate the Gini coefficient
    n = len(balances)
    gini = (2 * np.sum(np.arange(1, n + 1) * sorted_balances) - (n + 1) * np.sum(sorted_balances)) / (n * np.sum(sorted_balances))

    return gini


import numpy as np

def calculate_variance(balances):
    """Calculate the variance of a list of account balances."""
    balances = np.array(balances)
    balances_in_eth = [float(Web3.from_wei(balance, 'ether')) for balance in balances]
    variance = np.var(balances_in_eth)
    return variance


import numpy as np



run_challenge4()


