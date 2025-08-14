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


import requests

def get_eth_price():
    """Fetches the current price of ETH from CoinGecko API."""
    url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
    response = requests.get(url)
    data = response.json()
    eth_price = data["ethereum"]["usd"]
    return eth_price

current_ETH = get_eth_price() # Current ETH Price in dollars
print(f'1 ether is currently ${current_ETH}')

def dollar_to_GweiETH(dollars):
    Eth_dollar_unit = 1 / current_ETH
    ETH = dollars * Eth_dollar_unit # Number of ETH
    Gwei = ETH * 10**9 # Number of Gwei
    return ETH, Gwei

res_1dollar = dollar_to_GweiETH(1)
print(f'\n1 dollar is currently {res_1dollar[0]} ether \n1 dollar is currently {res_1dollar[1]} Gwei')


<marquee style='width: 30%; color: Coral;'><b>It Works!</b></marquee>


<head>
  <style>
    button {
      font-size: 2rem;        /*  1 rem is 16 pixels and 12 point */
      padding: 10px 20px;     /*  x padding and y padding         */
    }
  </style>
</head>
<body>
  <button id="snack" onclick="increment()">0</button>
  <script>
    let delta = 1;
    function increment() {
      const btn = document.getElementById('snack');        // our "snack" clicker!
      let count = parseInt(btn.innerText);                 // get the button number
      if (count >= 4.2 || count <= -4.2) { delta *= -1; }  // switch the direction
      btn.innerText = count + delta;                       // update with the current delta
    }
  </script>
</body>
</html>


<marquee style='width: 30%; color: Coral;'><b>The cookie clicke works realy nice. Might be an interesting addition to add buttons in future Fintech assignments!</b></marquee>


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

def total_dollars():
    total_in_eth = 0
    index = 0
    for account in w3.eth.accounts:
        balance_in_wei = w3.eth.get_balance(account)
        balance_in_ETH = Web3.from_wei(balance_in_wei, 'ether')
        total_in_eth += balance_in_ETH
        # print(f"Account #{index}  amount: {balance_in_ETH} ether")
        index += 1

    total_in_dollars = current_ETH * float(total_in_eth)
    print(f'There is a total of ${total_in_dollars} from all accounts\n')
    return total_in_dollars

total_dollars()



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




# Example of resetting the universe and running the results:
provider = Web3.EthereumTesterProvider()
w3 = Web3(provider)

GiniCoefficients, AllBalances = run_transactions()


'''TASK 1'''

import random
import time
import numpy as np
from eth_utils import ValidationError

def send_random_transaction(w3):
    # Choose random sender and receiver (ensuring they are different)
    accounts = w3.eth.accounts
    sender, receiver = random.sample(accounts, 2)

    # Choose random ether amount (between 0.01 and max available ether)
    ether_amount = random.uniform(0.01, 10)  # float

    # Create transaction dict
    transaction = {
        'from': sender,
        'to': receiver,
        'value': w3.to_wei(str(ether_amount), 'ether')
    }

    # Print what's happening
    print(f"{sender} is sending {ether_amount:.4f} ether to {receiver}")

    # Try to send the transaction
    try:
        tx_hash = w3.eth.send_transaction(transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        actual_gas_used = receipt['gasUsed']
        print(f"  ✓ Transaction sent! Hash: {tx_hash.hex()}\n")
        return actual_gas_used
    except ValidationError as e:
        print(f"  ✗ Transaction failed: {e}\n")
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}\n")

N = 10 # Number of transactions

def run_n_transactions(w3, N, delay_range=(2, 3)):
    total_gas_used = 0

    for i in range(N):
        print(f"--- Transaction {i+1}/{N} ---")
        gas_used = send_random_transaction(w3)
        total_gas_used += gas_used
        time.sleep(random.uniform(*delay_range))

    return total_gas_used
a = run_n_transactions(w3, N=5, delay_range=(2, 3))



'''TASK 2'''

def gini_coefficient(values):
    """Compute Gini coefficient for a list of values."""
    sorted_vals = np.sort(np.array(values))
    n = len(values)
    cumvals = np.cumsum(sorted_vals)
    gini = (n + 1 - 2 * np.sum(cumvals) / cumvals[-1]) / n
    return gini

N = 10

def run_marketplace(w3):
    total_gas = run_n_transactions(w3, N, delay_range=(2, 3))

    print("\n🔹 Final Account Balances:")
    balances = []
    for acct in w3.eth.accounts:
        bal = w3.from_wei(w3.eth.get_balance(acct), 'ether')
        balances.append(bal)
        print(f"{acct}: {bal:.4f} ETH")

    total_eth = sum(balances)
    variance = np.var(balances)
    gini = gini_coefficient(balances)

    print(f"\n🔸 Total Ether in System: {total_eth:.4f} ETH")
    print(f"🔸 Total Gas Used: {total_gas} gas")
    print(f"🔸 Variance in Balances: {variance:.6f}")
    print(f"🔸 Gini Coefficient: {gini:.4f}")

run_marketplace(w3)


'''TASK 3'''

N = 100

# Define new run N transactions function
def run_n_transactions(w3, N, delay_range=(2, 3)):
    import random
    import time

    gas_price = 1_000_000_000  # 1 Gwei
    total_gas_used = 0
    solvent_accounts = set(w3.eth.accounts)  # maintain only solvent accounts

    for i in range(N):
        if len(solvent_accounts) < 2:
            print("❗ Stopping: not enough solvent accounts.")
            break

        sender, receiver = random.sample(list(solvent_accounts), 2)
        sender_balance = w3.eth.get_balance(sender)
        ether_amount = random.uniform(0.01, 5)
        value_wei = w3.to_wei(str(ether_amount), 'ether')
        gas_fee = 21000 * gas_price
        total_needed = value_wei + gas_fee

        if total_needed > sender_balance:
            # Try to spend everything except gas
            value_wei = max(sender_balance - gas_fee, 0)
            if value_wei == 0:
                print(f"✗ Tx {i+1}: {sender[-4:]} is insolvent. Removing.")
                solvent_accounts.remove(sender)
                continue

        tx = {
            'from': sender,
            'to': receiver,
            'value': value_wei,
            'gas': 21000,
            'gasPrice': gas_price
        }

        try:
            tx_hash = w3.eth.send_transaction(tx)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            print(f"✓ Tx {i+1}: {sender[-4:]} → {receiver[-4:]} | Sent: {w3.from_wei(value_wei, 'ether'):.3f} ETH | Gas: {receipt['gasUsed']}")
            total_gas_used += receipt['gasUsed']
        except Exception as e:
            print(f"✗ Tx {i+1}: failed. {sender[-4:]} removed. Error: {e}")
            solvent_accounts.remove(sender)

        time.sleep(random.uniform(*delay_range))

    return total_gas_used

run_marketplace(w3)


'''TONTINE'''

import matplotlib.pyplot as plt
import numpy as np

def run_tontine(w3, tx_per_generation=10, delay_range=(0.1, 0.3)):
    generations = 0
    gini_history = []
    balance_history = []

    while True:
        generations += 1
        print(f"\n--- Generation {generations} ---")
        run_n_transactions(w3, tx_per_generation, delay_range)

        # Get current balances
        balances = [w3.from_wei(w3.eth.get_balance(acct), 'ether') for acct in w3.eth.accounts]
        balance_history.append(balances)
        gini_history.append(gini_coefficient(balances))

        # Check how many accounts still have positive balances (ignoring gas dust)
        positive = sum(b > 0.001 for b in balances)
        if positive <= 1:
            print("🏁 Tontine complete!")
            break

    return balance_history, gini_history


def plot_tontine(balance_history, gini_history, w3):
    balance_history = np.array(balance_history)

    # Plot balances
    plt.figure(figsize=(12, 5))
    for i in range(balance_history.shape[1]):
        plt.plot(balance_history[:, i], label=f"Acct {w3.eth.accounts[i][-4:]}")
    plt.title("Account Balances Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Balance (ETH)")
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()

    # Plot Gini
    plt.figure(figsize=(8, 4))
    plt.plot(gini_history, marker='o')
    plt.title("Gini Coefficient Over Time")
    plt.xlabel("Generation")
    plt.ylabel("Gini Coefficient")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


