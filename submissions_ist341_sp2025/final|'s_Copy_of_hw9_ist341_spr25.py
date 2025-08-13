###I used this instead of uploading straight onto colab so I can get a percentage
##this took an hour and a half and then crashed.. I think I need to find a smaller more managable dataset..

import pandas as pd
from google.colab import files
import io  # Add this import

# Step 1: Properly upload and read the file
uploaded = files.upload()

# Get the file data as bytes and decode it
file_name = next(iter(uploaded))  # Gets the first uploaded filename
file_data = uploaded[file_name].decode('utf-8')  # Decode the bytes

# Read CSV from the string data
df = pd.read_csv(io.StringIO(file_data))

# Step 2: Show all columns (to identify unwanted ones)
print("Original columns:\n", df.columns.tolist())

# Step 3: Select columns to KEEP (customize this!)
columns_to_keep = ['name', 'price', 'average_playtime', 'genres', 'release_date']

# Step 4: Create cleaned DataFrame
df_clean = df[columns_to_keep]

# Step 5: Verify results
print("\nCleaned columns:\n", df_clean.columns.tolist())
print("\nFirst 3 rows:\n", df_clean.head(3))

# Step 6: Save cleaned data
df_clean.to_csv('cleaned_games.csv', index=False)
files.download('cleaned_games.csv')


#
# Another notebook is equally ok!
#
#
#


###I need to find a more manageable dataset, and I think I would like to personalize or change the way I think about the data.
#Price vs playtime is a very basic dataset and I would like to be able to tell a story that is valuable to companies so perhaps sales or player retention
#would be an easier way to tell that story. Playtime can be skewed by 'AFK' players. Particularly, the price of games is going up and gamers are known to
#rage online. The idea of charging more for a game based on value and replayability is interesting. For me, games like Balatro can get 400 hours of entertainment for 20 dollars
#I will have to think about how to reframe the narrative and find manageable data that supports my ideas and interests. Once I get a clean dataset, hopefully I can get some velocity


#
# hw9, problem 1 NNETS!
#
#   including _both_ clasification + regression for iris-species modeling
#


# libraries...
import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)


# let's read in our flower data...
#
# iris_cleaned.csv and hw4pr1iris_cleaner.ipynb should be in this folder
#
filename = 'iris_cleaned.csv'
df_tidy = pd.read_csv(filename)      # encoding = "utf-8", "latin1"
print(f"{filename} : file read into a pandas dataframe.")


#
# different version vary on how to see all rows (adapt to suit your system!)
#
print(f"df_tidy.shape is {df_tidy.shape}\n")
df_tidy.info()  # prints column information
df_tidy


#
# All of the columns need to be numeric, we'll drop irisname
ROW = 0
COLUMN = 1
df_model1 = df_tidy.drop( 'irisname', axis=COLUMN )
df_model1


#
# once we have all the columns we want, let's create an index of their names...

#
# Let's make sure we have all of our helpful variables in one place
#       To be adapted if we drop/add more columns...
#

#
# let's keep our column names in variables, for reference
#
COLUMNS = df_model1.columns            # "list" of columns
print(f"COLUMNS is {COLUMNS}\n")
  # It's a "pandas" list, called an Index
  # use it just as a Python list of strings:
print(f"COLUMNS[0] is {COLUMNS[0]}\n")

# let's create a dictionary to look up any column index by name
COL_INDEX = {}
for i, name in enumerate(COLUMNS):
    COL_INDEX[name] = i  # using the name (as key), look up the value (i)
print(f"COL_INDEX is {COL_INDEX}\n\n")


#
# and our "species" names
#

# all of scikit-learn's ML routines need numbers, not strings
#   ... even for categories/classifications (like species!)
#   so, we will convert the flower-species to numbers:

SPECIES = ['setosa','versicolor','virginica']   # int to str
SPECIES_INDEX = {'setosa':0,'versicolor':1,'virginica':2}  # str to int

# Let's try it out...
for name in SPECIES:
    print(f"{name} maps to {SPECIES_INDEX[name]}")


#
# We _could_ reweight our columns...
# For exmaple, if petalwid were "worth" 20x more than the others?
#

# df_model1['petalwid'] *= 20
# df_model1

#
# But, with NNets, the whole goal of the _network_ is to weight each feature itself!
#
#      Let's see how it does:
#


#
# let's convert our dataframe to a numpy array, named A
#
A = df_model1.to_numpy()
A = A.astype('float64')    # many types:  www.tutorialspoint.com/numpy/numpy_data_types.htm
print(A)


#
# nice to have NUM_ROWS and NUM_COLS around
#
NUM_ROWS, NUM_COLS = A.shape
print(f"\nThe dataset has {NUM_ROWS} rows and {NUM_COLS} cols")


# let's use all of our variables, to reinforce that we have
# (1) names...
# (2) access and control...

# choose a row index, n:
n = 42
print(f"flower #{n} is {A[n]}")

for i in range(len(COLUMNS)):
    colname = COLUMNS[i]
    value = A[n][i]
    print(f"  Its {colname} is {value}")

species_index = COL_INDEX['irisnum']
species_num = int(round(A[n][species_index]))
species = SPECIES[species_num]
print(f"  Its species is {species} (i.e., {species_num})")


print("+++ Start of data definitions +++\n")

#
# we could do this at the data-frame level, too!
#

X_all = A[:,0:4]  # X (features) ... is all rows, columns 0, 1, 2, 3
y_all = A[:,4]    # y (labels) ... is all rows, column 4 only

print(f"y_all (just the labels/species)   are \n {y_all}")
print(f"X_all (just the features, first few rows) are \n {X_all[0:5]}")


#
# we can scramble the data, to remove (potential) dependence on its ordering:
#
indices = np.random.permutation(len(y_all))  # indices is a permutation-list

# we scramble both X and y, necessarily with the same permutation
X_all = X_all[indices]              # we apply the _same_ permutation to each!
y_all = y_all[indices]              # again...
print(f"The scrambled labels/species are \n {y_all}")
print(f"The corresponding data rows are \n {X_all[0:5]}")


#
# We next separate into test data and training data ...
#    + We will train on the training data...
#    + We will _not_ look at the testing data to build the model
#
# Then, afterward, we will test on the testing data -- and see how well we do!
#

#
# a common convention:  train on 80%, test on 20%    Let's define the TEST_PERCENT
#

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2)

print(f"training with {len(y_train)} rows;  testing with {len(y_test)} rows\n" )

print(f"Held-out data... (testing data: {len(y_test)})")
print(f"y_test: {y_test}\n")
print(f"X_test (few rows): {X_test[0:5,:]}")  # 5 rows
print()
print(f"Data used for modeling... (training data: {len(y_train)})")
print(f"y_train: {y_train}\n")
print(f"X_train (few rows): {X_train[0:5,:]}")  # 5 rows


#
# for NNets, it's important to keep the feature values near 0, say -1. to 1. or so
#    This is done through the "StandardScaler" in scikit-learn
#
from sklearn.preprocessing import StandardScaler

USE_SCALER = True   # this variable is important! It tracks if we need to use the scaler...

# we "train the scaler"  (computes the mean and standard deviation)
if USE_SCALER == True:
    scaler = StandardScaler()
    scaler.fit(X_train)  # Scale with the training data! ave becomes 0; stdev becomes 1
else:
    # this one does no scaling!  We still create it to be consistent:
    scaler = StandardScaler(copy=True, with_mean=False, with_std=False) # no scaling
    scaler.fit(X_train)  # still need to fit, though it does not change...

scaler   # is now defined and ready to use...

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Here are our scaled training and testing sets:

X_train_scaled = scaler.transform(X_train) # scale!
X_test_scaled = scaler.transform(X_test) # scale!

y_train_scaled = y_train  # the predicted/desired labels are not scaled
y_test_scaled = y_test  # not using the scaler




#
# let's create a table for showing our data and its predictions...
#
def ascii_table(X,y,scaler_to_invert=None):
    """ print a table of binary inputs and outputs """
    if scaler_to_invert == None:  # don't use the scaler
        X = X
    else:
        X = scaler_to_invert.inverse_transform(X)
    print(f"{'input ':>58s} -> {'pred':<5s} {'des.':<5s}")
    for i in range(len(y)):
        # whoa! serious f-string formatting:
        print(f"{X[i,0:4]!s:>58s} -> {'?':<5s} {y[i]:<5.2f}")   # !s is str ...
    print()

# to show the table with the scaled data:
ascii_table(X_train_scaled[0:5,:],y_train_scaled[0:5],None)

# to show the table with the original data:
ascii_table(X_train_scaled[0:5,:],y_train_scaled[0:5],scaler)


from sklearn.neural_network import MLPClassifier

#
# Here's where you can change the number of hidden layers
# and number of neurons!
#
nn_classifier = MLPClassifier(hidden_layer_sizes=(6,7),  # 3 input -> 6 -> 7 -> 1 output
                    max_iter=500,      # how many times to train
                    activation="tanh", # the "activation function" input -> output
                    solver='sgd',      # the algorithm for optimizing weights
                    verbose=True,      # False to "mute" the training
                    shuffle=True,      # reshuffle the training epochs?
                    random_state=None, # set for reproduceability
                    learning_rate_init=.1,       # learning rate: % of error to backprop
                    learning_rate = 'adaptive')  # soften feedback as it converges

# documentation:
# scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
#     Try verbose / activation "relu" / other network sizes ...

print("\n\n++++++++++  TRAINING:  begin  +++++++++++++++\n\n")
nn_classifier.fit(X_train_scaled, y_train_scaled)
print("\n++++++++++  TRAINING:   end  +++++++++++++++")
print(f"The analog prediction error (the loss) is {nn_classifier.loss_}")


#
# how did it do on the testing data?
#

def ascii_table_for_classifier(Xsc,y,nn,scaler):
    """ a table including predictions using nn.predict """
    predictions = nn.predict(Xsc)            # all predictions
    prediction_probs = nn.predict_proba(Xsc) # all prediction probabilities
    Xpr = scaler.inverse_transform(Xsc)      # Xpr is the "X to print": unscaled data!
    # count correct
    num_correct = 0
    # printing
    print(f"{'input ':>28s} -> {'pred':^6s} {'des.':^6s}")
    for i in range(len(y)):
        pred = predictions[i]
        pred_probs = prediction_probs[i,:]
        desired = y[i]
        if pred != desired: result = "  incorrect: " + str(pred_probs)
        else: result = "  correct"; num_correct += 1
        # Xpr = Xsc  # if you want to see the scaled versions
        print(f"{Xpr[i,0:4]!s:>28s} -> {pred:^6.0f} {desired:^6.0f} {result:^10s}")
    print(f"\ncorrect predictions: {num_correct} out of {len(y)}")

#
# let's see how it did on the test data (also the training data!)
#
ascii_table_for_classifier(X_test_scaled,
                           y_test_scaled,
                           nn_classifier,
                           scaler)



#
# We don't usually look inside the NNet, but we can: it's open-box modeling...
#
if True:  # do we want to see all of the parameters?
    nn = nn_classifier  # less to type?
    print("\n\n+++++ parameters, weights, etc. +++++\n")
    print(f"\nweights/coefficients:\n")
    for wts in nn.coefs_:
        print(wts)
    print(f"\nintercepts: {nn.intercepts_}")
    print(f"\nall parameters: {nn.get_params()}")


#
# we have a predictive model!  Let's try it out...
#

def make_prediction( Features, nn, scaler ):
    """ uses nn for predictions """
    print("input features are", Features)
    #  we make sure Features has the right shape (list-of-lists)
    row = np.array( [Features] )  # makes an array-row
    row = scaler.transform(row)   # scale according to scaler
    print("nn.predict_proba == ", nn.predict_proba(row))   # probabilities of each
    prediction = nn.predict(row)  # max!
    return prediction

# our features -- note that the inputs don't have to be bits!
Features = [ 2.7, 2.1, 5.6, 0.4 ]      # whatever we'd like to test
prediction = make_prediction(Features, nn_classifier, scaler)
print(f"prediction: {prediction}")     # takes the max (nice to see them all!)


#
# What shall we predict today?
print(COL_INDEX)
print(COLUMNS)

# Let's first predict sepal length ('sepallen', column index 0)


#
# Here we set up for a regression model that will predict 'sepallen'  (column index 0)  using
#

#   sepal width   'sepalwid' (column index 1)
#   petal length  'petallen' (column index 2)
#   petal width   'petalwid' (column index 3)
#     and
#   species       'irisnum'  (column index 4)

print("+++ Start of data-assembly for feature-regression! +++\n")
# construct the correct X_all from the columns we want
# we use np.concatenate to combine parts of the dataset to get all-except-column 0:
X_all = np.concatenate( (A[:,0:0], A[:,1:]),axis=1)  # columns 1, 2, 3, and 4

# if we wanted all-except-column 1:   X_all = np.concatenate( (A[:,0:1], A[:,2:]),axis=1)  # columns 0, 2, 3, and 4
# if we wanted all-except-column 2:   X_all = np.concatenate( (A[:,0:2], A[:,3:]),axis=1)  # columns 0, 1, 3, and 4
# if we wanted all-except-column 3:   X_all = np.concatenate( (A[:,0:3], A[:,4:]),axis=1)  # columns 0, 1, 2, and 4
# if we wanted all-except-column 4:   X_all = np.concatenate( (A[:,0:4], A[:,5:]),axis=1)  # columns 0, 1, 2, and 3
# slicing is forgiving...


y_all = A[:,0]             # y (labels) ... is all of column 0, sepallen (sepal length)  Re-index, as needed...
print(f"y_all is \n {y_all}")
print(f"X_all (just features: first few rows) is \n {X_all[:5,:]}")



#
# we scramble the data, to give a different TRAIN/TEST split each time...
#
indices = np.random.permutation(len(y_all))  # indices is a permutation-list

# we scramble both X and y, necessarily with the same permutation
X_all = X_all[indices]              # we apply the _same_ permutation to each!
y_all = y_all[indices]              # again...
print("label-_values_\n",y_all)
print("\nfeatures (a few)\n", X_all[:5,:])


#
# We next separate into test data and training data ...
#    + We will train on the training data...
#    + We will _not_ look at the testing data to build the model
#
# Then, afterward, we will test on the testing data -- and see how well we do!
#

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2)

print(f"training with {len(y_train)} rows;  testing with {len(y_test)} rows\n" )

print(f"Held-out data... (testing data: {len(y_test)})")
print(f"y_test: {y_test}\n")
print(f"X_test (few rows): {X_test[0:5,:]}")  # 5 rows
print()
print(f"Data used for modeling... (training data: {len(y_train)})")
print(f"y_train: {y_train}\n")
print(f"X_train (few rows): {X_train[0:5,:]}")  # 5 rows


#
# for NNets, it's important to keep the feature values near 0, say -1. to 1. or so
#    This is done through the "StandardScaler" in scikit-learn
#
from sklearn.preprocessing import StandardScaler

USE_SCALER = True   # this variable is important! It tracks if we need to use the scaler...

# we "train the scaler"  (computes the mean and standard deviation)
if USE_SCALER == True:
    scaler = StandardScaler()
    scaler.fit(X_train)  # Scale with the training data! ave becomes 0; stdev becomes 1
else:
    # this one does no scaling!  We still create it to be consistent:
    scaler = StandardScaler(copy=True, with_mean=False, with_std=False) # no scaling
    scaler.fit(X_train)  # still need to fit, though it does not change...

scaler   # is now defined and ready to use...

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Here are our scaled training and testing sets:

X_train_scaled = scaler.transform(X_train) # scale!
X_test_scaled = scaler.transform(X_test) # scale!

y_train_scaled = y_train  # the predicted/desired labels are not scaled
y_test_scaled = y_test  # not using the scaler

# reused from above - seeing the scaled data
ascii_table(X_train_scaled[0:5,:],y_train_scaled[0:5],None)

# reused from above - seeing the unscaled data (inverting the scaler)
ascii_table(X_train_scaled[0:5,:],y_train_scaled[0:5],scaler)


#
# MLPRegressor predicts _floating-point_ outputs
#

from sklearn.neural_network import MLPRegressor

nn_regressor = MLPRegressor(hidden_layer_sizes=(6,7),
                    max_iter=200,          # how many training epochs
                    activation="tanh",     # the activation function
                    solver='sgd',          # the optimizer
                    verbose=True,          # do we want to watch as it trains?
                    shuffle=True,          # shuffle each epoch?
                    random_state=None,     # use for reproducibility
                    learning_rate_init=.1, # how much of each error to back-propagate
                    learning_rate = 'adaptive')  # how to handle the learning_rate

print("\n\n++++++++++  TRAINING:  begin  +++++++++++++++\n\n")
nn_regressor.fit(X_train_scaled, y_train_scaled)
print("++++++++++  TRAINING:   end  +++++++++++++++")

print(f"The (squared) prediction error (the loss) is {nn_regressor.loss_}")
print(f"And, its square root: {nn_regressor.loss_ ** 0.5}")


#
# Note that square-root of the mean-squared-error is an "expected error" in predicting our feature
#
# It's a common measure of expected error, but only one.
#


#
# how did it do? Try out the TEST data...
#

def ascii_table_for_regressor(Xsc,y,nn,scaler):
    """ a table including predictions using nn.predict """
    predictions = nn.predict(Xsc) # all predictions
    Xpr = scaler.inverse_transform(Xsc)  # Xpr is the "X to print": unscaled data!
    # measure error
    error = 0.0
    # printing
    print(f"{'input ':>28s} ->  {'pred':^6s}  {'des.':^6s}  {'absdiff':^10s}")
    for i in range(len(y)):
        pred = predictions[i]
        desired = y[i]
        result = abs(desired - pred)
        error += result
        # Xpr = Xsc   # if you'd like to see the scaled values
        print(f"{Xpr[i,0:4]!s:>28s} ->  {pred:<+6.3f}  {desired:<+6.3f}  {result:^10.3f}")

    print("\n" + "+++++   +++++      +++++   +++++   ")
    print(f"average abs error: {error/len(y)}")
    print("+++++   +++++      +++++   +++++   ")

#
# let's see how it did on the test data (also the training data!)
#
ascii_table_for_regressor(X_test_scaled,
                          y_test_scaled,
                          nn_regressor,
                          scaler)   # this is our own f'n, above



"""
Your tasks!

Just as above, find the average abs. error in the _other_ three botanical features:
  + sepalwid
  + petallen
  + petalwid
  + (above is sepallen, already complete!) Use as a template!

Copy-and-paste option:
  + copy-and-paste a lot!
  + You will feel like regressing is second nature (this isn't a bad thing!)
  + You might feel you've regressed... a challenge is the looping approach, instead:

Looping option:
  + run a loop over the columns... (see the concatenate call, above)
  + When you use concatenate, be sure all are _slices_, not single columns

Then, include a text (or markdown) cell with your _four_ av. abs. errors on the TEST set
  + this is one average absolute error for each of the four botanical features

Just as some features are more important than others,
  + so, too are some features more _predictable_ than others...

Not required...  but interesting:
  How much do different network-shapes matter here?

Not required... but also interesting:
  You could _then_ drop the species column...

  ... in order to estimate how much "value" is added, by knowing the flower-species,
  when predicting each botanical measurement...

The truth is, these are not part of this hw. BUT, they're here as questions,
because they are all great "final-project-type" explorations -- even better,
if they use your own data!

"""



def correlation(x, y):
    if len(x) != len(y):
        raise ValueError("Lists must have the same length")

    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    denominator_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5

    if denominator_x == 0 or denominator_y == 0:
        raise ValueError("One of the lists has zero variance")

    return numerator / (denominator_x * denominator_y)





a = [1, 2, 3, 4, 5]
b = [2, 4, 6, 8, 10]

print(correlation(a, b))


import random

# Generate 3 stocks, each with 10 values
stock_values = [
    [round(random.uniform(50, 150), 2) for _ in range(10)]
    for _ in range(3)
]

print(stock_values)


