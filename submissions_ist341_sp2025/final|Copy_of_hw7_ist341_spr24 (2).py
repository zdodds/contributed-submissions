#
# Let's practice the collapsing! And f-strings:
#

x = 42
ascii_emoji = "<3"

print(f"x is {x}.   We {ascii_emoji} f-strings!")



#
# iris cleaner:  data-cleaning for iris modeling and classification
#

#
# Here, our goal is to
# [1] look over the iris.csv data...
# [2] clean it up, removing rows and columns we don't want to use
# [3] save the "cleaned-up data" to a new filename, iris_cleaned.csv

#
# Then, we can use iris_cleaned.csv for _ALL_ of our iris-modeling from here...
#


#
# Side note only!
# # don't worry about this cell - it's just an example of a _SILLY_ data model
# # DON'T copy this cell over when you model the births-data or the digits-data
#
# It's here, because it's worth noting that we don't _need_ any data at all to create a predictive model!
#
# # Here is a model that is half hand-built and half random. No data is used!
#
import random

def predictive_model( Features ):
    """ input: a list of four features
                [ sepallen, sepalwid, petallen, petalwid ]
        output: the predicted species of iris, from
                  setosa (0), versicolor (1), virginica (2)
    """
    [ sepallen, sepalwid, petallen, petalwid ] = Features # unpacking!

    if petalwid < 1.0:
        return 'setosa (0)'
    else:
        return random.choice( ['versicolor (1)', 'virginica (2)'] )

#
# Try it!
#
# Features = eval(input("Enter new Features: "))
#
Features = [ 4.6, 3.6, 3.0, 1.92 ]
result = predictive_model( Features )
print(f"from Features {Features},  I predict...   {result} ")


#
# (Next, let's explore how we _can_ use data to do better... :-)
#


# libraries!
import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)


# let's read in our flower data...
#
# for read_csv, use header=0 when row 0 is a header row
#
filename = 'iris.csv'
df = pd.read_csv(filename)        # encoding="utf-8" et al.
print(f"{filename} : file read into a pandas dataframe.")


#
# a dataframe is a "spreadsheet in Python"   (seems to have an extra column!)
#
# let's view it!



#
# Looking at the result, above, we see some things that need to be "tidied":
#
# [1] there's an extra column (holding the reference url)
# [2] there are some flowers not in our three speciesL setosa, versicolor, virginica
# [3] there is a flower without a species name (irisname)
# [4] this is a virginica flower without a petallen
#


#
# let's look at the dataframe's "info":
df.info()


# Let's look at the dataframe's columns -- and remind ourselves of for loops!
for column_name in df.columns:
    print(f"{column_name =}")


# we can drop a series of data (a row or a column)
# the dimensions each have a numeric value, row~0, col~1, but let's use readable names we define:
ROW = 0
COLUMN = 1

df_clean1 = df.drop('adapted from https://en.wikipedia.org/wiki/Iris_flower_data_set', axis=COLUMN)
df_clean1

# df_clean1 is a new dataframe, without that unwanted column


df_clean2 = df_clean1


# and, let's drop the unwanted rows:
ROW = 0
COLUMN = 1

df_clean2 = df_clean1.drop([142,143,144], axis=ROW)
df_clean2


#
# let's re-look at our cleaned-up dataframe's info:
#
df_clean2.info()
#
# notice that the non-null count is _different_ across the features...
#


#
# let's drop _all_ rows with data that is missing/NaN (not-a-number)
df_clean3 = df_clean2.dropna()  # drop na rows (NaN, not-a-number)
df_clean3.info()  # print the info, and
# let's see the whole table, as well:
df_clean3

# Tidy!  Our data is ready!


#
# let's keep our column names in variables, for reference
#
COLUMNS = df_clean1.columns            # "list" of columns
print(f"COLUMNS is {COLUMNS}")
  # It's a "pandas" list, called an Index
  # use it just as a Python list of strings:
print(f"COLUMNS[0] is {COLUMNS[0]}\n")

# let's create a dictionary to look up any column index by name
COL_INDEX = {}
for i, name in enumerate(COLUMNS):
    COL_INDEX[name] = i  # using the name (as key), look up the value (i)
print(f"COL_INDEX is {COL_INDEX}")
print(f"COL_INDEX[ 'petallen' ] is {COL_INDEX[ 'petallen' ]}")



# all of scikit-learn's ML routines need numbers, not strings
#   ... even for categories/classifications (like species!)
#   so, we will convert the flower-species to numbers

#
# First, let's map our different species to numeric values:

SPECIES = ['setosa','versicolor','virginica']   # int to str
SPECIES_INDEX = {'setosa':0,'versicolor':1,'virginica':2}  # str to int

def convert_species(speciesname):
    """ return the species index (a unique integer/category) """
    #print(f"converting {speciesname}...")
    return SPECIES_INDEX[speciesname]

# Let's try it out...
for name in SPECIES:
    print(f"{name} maps to {convert_species(name)}")


convert_species( 'virginica')  # try converting from string to index!


# Convert the other direction, from integer index to species name
SPECIES[2]


#
# we can "apply" to a whole column and create a new column
#   it may give a warning, but this is ok...
#

df_clean4 = df_clean3.copy()  # copy everything AND...

# add a new column, 'irisnum'
df_clean4['irisnum'] = df_clean3['irisname'].apply(convert_species)

# let's see...
df_clean4


#
# different version vary on how to see all rows (adapt to suit your system!)
#
# pd.options.display.max_rows = 150   # None for no limit; default: 10
# pd.options.display.min_rows = 150   # None for no limit; default: 10
# pd.options.display.max_rows = 10   # None for no limit; default: 10
# pd.options.display.min_rows = 10   # None for no limit; default: 10
for row in df_clean4.itertuples():
    print(row)


#
# let's call it df_tidy
#
df_tidy =  df_clean4



#
# That's it!  Then, and write it out to iris_cleaned.csv

# We'll construct the new filename:
old_basename = filename[:-4]                      # remove the ".csv"
cleaned_filename = old_basename + "_cleaned.csv"  # name-creating
print(f"cleaned_filename is {cleaned_filename}")

# Now, save
df_tidy.to_csv(cleaned_filename, index_label=False)  # no "index" column...


#
# Let's make sure this worked, by re-reading in the data...
#

# let's re-read that file and take a look...
#
# for read_csv, use header=0 when row 0 is a header row
#
df_tidy_reread = pd.read_csv(cleaned_filename)   # encoding="utf-8" et al.
print(f"{filename} : file read into a pandas dataframe.")
df_tidy_reread


#
# Let's make sure we have all of our helpful variables in one place
#
#   This will be adapted if we drop/add more columns...
#

#
# let's keep our column names in variables, for reference
#
COLUMNS = df_tidy.columns            # "list" of columns
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

def convert_species(speciesname):
    """ return the species index (a unique integer/category) """
    #print(f"converting {speciesname}...")
    return SPECIES_INDEX[speciesname]

# Let's try it out...
for name in SPECIES:
    print(f"{name} maps to {convert_species(name)}")


#
# That's it!  Welcome to the world of data-cleaning workflows!!
#
#             Our prediction?  You'll be headed to the "modeler" next!
#

#
# And, the rest of the hw is to run more ML workflows:   (1) Births, (2) Digits, (3) Titanic, (ec) Housing, ...
#


#
# iris modeler:  iris clasification via nearest neighbors
#


# libraries!
import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)


# let's read in our flower data...
#
# for read_csv, use header=0 when row 0 is a header row
#
cleaned_filename = "iris_cleaned.csv"
df_tidy = pd.read_csv(cleaned_filename)   # encoding="utf-8" et al.
print(f"{cleaned_filename} : file read into a pandas dataframe.")
df_tidy


#
# here's how to view every single row at once -- it's a lot!
for row in df_tidy.itertuples():
    print(row)


#
# Let's drop the columns [features] we don't want/need
#                or that we _shouldn't_ have...!
#

# First, look at the info:
df_tidy.info()


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
# What if petalwid is "worth" 20x more than the others?
#
df_model1['petalwid'] *= 20
df_model1



# Until we have more insight, this is arbitrary at best and data-rigging, at worst.
# So, let's set it back...
df_model1['petalwid'] /= 20
df_model1



#
A = df_model1.to_numpy()    # yields the underlying numpy array
A = A.astype('float64')     # make sure it's all floating point  (www.tutorialspoint.com/numpy/numpy_data_types.htm)
print(A[0:5])               # A is too big, let's just sanity-check



#
# Also, nice to have NUM_ROWS and NUM_COLS around
#
NUM_ROWS, NUM_COLS = A.shape
print(f"\nThe dataset has {NUM_ROWS} rows and {NUM_COLS} cols")


df_tidy.info()


print("+++ Start of data definitions +++\n")

#
# we could do this at the data-frame level, too!
#

X_all = A[:,0:4]  # X (features) ... is all rows, columns 0, 1, 2, 3
y_all = A[:,4]    # y (labels) ... is all rows, column 4 only

print(f"y_all (just the labels/species)   are \n {y_all}")
print(f"X_all (just the features - a few) are \n {X_all[0:5]}")



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

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2) # random_state=42

print(f"training with {len(y_train)} rows;  testing with {len(y_test)} rows\n" )

#
# Let's print the TRAINING data
#
print("+++++")
print(f"Held-out data... (testing data: {len(y_test)} rows)")
print("+++++\n")
print(f"y_test: {y_test}")
print(f"X_test (first few rows): {X_test[0:5,:]}")  # 5 rows
print()


#
# Let's print some of the TRAINING data
#

print("+++++")
print(f"Data used for modeling... (training data: {len(y_train)} rows)")
print("+++++\n")
print(f"y_train: {y_train}")
print(f"X_train (first few rows): {X_train[0:5,:]}")  # 5 rows


#
# +++ This is the "Model-building and Model-training Cell"
#
# Create a kNN model and train it!
#
from sklearn.neighbors import KNeighborsClassifier

k = 84   # we don't know what k to use, so we guess!  (this will _not_ be a good value)
knn_model = KNeighborsClassifier(n_neighbors=k)       # here, k is the "k" in kNN

# we train the model (it's one line!)
knn_model.fit(X_train, y_train)                              # yay!  trained!
print("Created and trained a knn classifier with k =", k)


#
# +++ This cell is our "Model-testing Cell"
#
# Now, let's see how well our model does on our "held-out data" (the testing data)
#

# We run our test set:

# the function knn_model.predict is the instantiation of our model
# it's what runs the k-nearest-neighbors algorithm:
predicted_labels = knn_model.predict(X_test)      # THIS IS THE KEY LINE:  predict
actual_labels = y_test

# Let's print them so we can compare...
print("Predicted labels:", predicted_labels)
print("Actual  labels  :", actual_labels)

# And, some overall results
num_correct = sum(predicted_labels == actual_labels)
total = len(actual_labels)
print(f"\nResults on test set:  {num_correct} correct out of {total} total.")


#
# Let's print things in a vertical table
#

def compare_labels(predicted_labels, actual_labels):
    """ a more neatly formatted comparison """
    NUM_LABELS = len(predicted_labels)
    num_correct = 0

    print()
    print(f'row {"#":>3s} : {"predicted":>12s} {"actual":<12s}   {"result"}')

    for i in range(NUM_LABELS):
        p = int(round(predicted_labels[i]))         # round protects from fp error
        a = int(round(actual_labels[i]))
        result = "incorrect"
        if p == a:  # if they match,
            result = ""       # no longer incorrect
            num_correct += 1  # and we count a match!

        print(f"row {i:>3d} : {SPECIES[p]:>12s} {SPECIES[a]:<12s}   {result}")

    print()
    print("Correct:", num_correct, "out of", NUM_LABELS)
    return num_correct

# let's try it out!
compare_labels(predicted_labels,actual_labels)


#
# Ok!  We have our knn model, we could just use it...

# data-driven predictive model (k-nearest-neighbor), using scikit-learn

# warning: this model has not yet been tuned to its "best k"
#
def predictive_model( Features ):
    """ input: a list of four features
                [ sepallen, sepalwid, petallen, petalwid ]
        output: the predicted species of iris, from
                  setosa (0), versicolor (1), virginica (2)
    """
    our_features = np.asarray([Features])                 # extra brackets needed
    predicted_species = knn_model.predict(our_features)   # PREDICT!

    predicted_species = int(round(predicted_species[0]))  # unpack one element
    name = SPECIES[predicted_species]                     # look up the species
    return name

#
# Try it!
#
# Features = eval(input("Enter new Features: "))
#
Features = [6.7,3.3,4.7,0.1]            # [5.8,2.7,4.1,1.0] [4.6,3.6,3.0,2.2] [6.7,3.3,5.7,2.1]
result = predictive_model( Features )
print(f"I predict {result} from Features {Features}")


#
# Except, we didn't really explore whether this was the BEST model we could build...
#
#
# We used k = 84  (a neighborhood size of 84 flowers)
# In a dataset of only 140ish flowers, with three species, this is a _bad_ idea!
#
# Perhaps we should try ALL the neighborhood sizes in their own TRAIN/TEST split
# and see which neighborhood size works the best, for irises, at least...
#
# This is "cross validation" ...
#


#
# Here, we use "cross validation" to find the "best" k...
#

from sklearn.model_selection import cross_val_score

#
# cross-validation splits the training set into two pieces:
#   + model-building and model-validation. We'll use "build" and "validate"
#
best_k = 84  # Not correct!
best_accuracy = 0.0  # also not correct...
all_accuracies = []

# Note that we are cross-validating using only our TEST data!
for k in range(1,85):
    knn_cv_model = KNeighborsClassifier(n_neighbors=k)   # build a knn_model for every k
    cv_scores = cross_val_score( knn_cv_model, X_train, y_train, cv=5 )  # cv=5 means 80/20
    this_cv_accuracy = cv_scores.mean()               # mean() is numpy's built-in average function
    print(f"k: {k:2d}  cv accuracy: {this_cv_accuracy:7.4f}")
    all_accuracies += [this_cv_accuracy]

    if this_cv_accuracy > best_accuracy:  # is this one better?
        best_accuracy = this_cv_accuracy  # track the best accuracy
        best_k = k                        # with the best k


# use best_k!
print(f"best_k = {best_k}   yields the highest average cv accuracy.")  # print the best one



### Let's see all the accuracies!

import pandas as pd
# Let's create a pandas dataframe out of the above cell's data
crossvalidation_df = pd.DataFrame( {"k_value":np.asarray(range(1,84+1)),
                                    "accuracy":np.asarray(all_accuracies)}
                                    )

import seaborn as sns
sns.set_theme(style="darkgrid")
# Plot the responses for different events and regions
sns.lineplot(x="k_value", y="accuracy",  #  hue="region", style="event",
             data=crossvalidation_df)


#
# With the best k, we build and train a new model:
#
# Now using best_k instead of the original, randomly-guessed value:
#
best_k = best_k   # not needed, but nice
from sklearn.neighbors import KNeighborsClassifier
knn_model_tuned = KNeighborsClassifier(n_neighbors=best_k)   # here, we use the best_k!

# we train the model (one line!)
knn_model_tuned.fit(X_train, y_train)                              # yay!  trained!
print(f"Created + trained a knn classifier, now tuned with a (best) k of {best_k}")

# How does it do?!  The next cell will show...


#
# Re-create and re-run the  "Model-testing Cell"     How does it do with best_k?!
#
predicted_labels = knn_model_tuned.predict(X_test)
actual_labels = y_test

# Let's print them so we can compare...
print("Predicted labels:", predicted_labels)
print("Actual labels:", actual_labels)

# And, the overall results
num_correct = sum(predicted_labels == actual_labels)
total = len(actual_labels)
print(f"\nResults on test set:  {num_correct} correct out of {total} total.\n\n")

# Plus, we'll print our nicer table...
compare_labels(predicted_labels,actual_labels)


#
# Ok!  We tuned our knn modeling to use the "best" value of k...
#
# And, we should now use ALL available data to train our final predictive model:
#
knn_model_final = KNeighborsClassifier(n_neighbors=best_k)     # here, we use the best_k
knn_model_final.fit(X_all, y_all)                              # KEY DIFFERENCE:  we use ALL the data!
print(f"Created + trained a 'final' knn classifier, with a (best) k of {best_k}")


#
# final predictive model (k-nearest-neighbor), with tuned k + ALL data incorporated
#

def predictive_model( Features, Model ):
    """ input: a list of four features
                [ sepallen, sepalwid, petallen, petalwid ]
        output: the predicted species of iris, from
                  setosa (0), versicolor (1), virginica (2)
    """
    our_features = np.asarray([Features])                 # extra brackets needed
    predicted_species = Model.predict(our_features)       # The model's prediction!
    predicted_species = int(round(predicted_species[0]))  # unpack the extra brackets
    return predicted_species

#
# Try it!
#
# Features = eval(input("Enter new Features: "))
#
# Features = [6.7,3.3,5.7,0.1]  # [5.8,2.7,4.1,1.0] [4.6,3.6,3.0,2.2] [6.7,3.3,5.7,2.1]

LoF = [
[4.8, 3.1, 1.6, 0.2 ],
[5.7, 2.9, 4.2, 1.3 ],
[5.8, 2.7, 5.1, 1.9 ],
[5.2, 4.1, 1.5, 0.1 ],
[5.4, 3.4, 1.5, 0.4 ],
[5.1, 2.5, 3.0, 1.1 ],
[6.2, 2.9, 4.3, 1.3 ],
[6.3, 3.3, 6.0, 2.5 ],
[5.7, 2.8, 4.1, 1.3 ],
]

# LoF =  [ [0.1,7.2,4.2,1.042] ]

# run on each one:
for Features in LoF:
    predicted_species = predictive_model( Features, knn_model_final )  # pass in the model, too!
    name = SPECIES[predicted_species]
    print(f"I predict {name} from the features {Features}")    # Answers in the assignment...


# we can only plot 2 dimensions at a time!
# These two will be our constants:
sepallen = 5.0
sepalwid = 3.0

VERTICAL = np.arange(0,10,.1) # array of vertical input values
HORIZONT = np.arange(0,10,.1) # array of horizontal input values
PLANE = np.zeros( (len(HORIZONT),len(VERTICAL)) ) # the output array

row = 0
col = 0
for petallen in VERTICAL: # for every sepal length
  for petalwid in HORIZONT: # for every sepal width
    Features = [ sepallen, sepalwid, petallen, petalwid ]
    output = predictive_model(Features,knn_model_final)
    #print(f"Input {Features} Output: {output}")
    PLANE[row,col] = output
    col += 1
  col = 0
  row += 1
  print(".", end="")  # so we know it's running
  if row % 42 == 0: print() # same...

print("\n", PLANE[0:3,0:3]) # small bit of the upper-left corner



# prompt: please plot the above heatmap, with 1/4 as many axis labels

# Assuming 'PLANE', 'VERTICAL', and 'HORIZONT' are defined as in the original code

# Create a new figure and axes
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 8))

# Create the heatmap
im = ax.imshow(PLANE, cmap="viridis", extent=[HORIZONT.min(), HORIZONT.max(), VERTICAL.min(), VERTICAL.max()], origin="lower", aspect="auto")

# Set axis labels and ticks
ax.set_xlabel("petalwid", fontsize=14)
ax.set_ylabel("petallen", fontsize=14)

# Calculate the indices for reduced ticks and labels
reduced_tick_indices = np.arange(0, len(HORIZONT), len(HORIZONT)//8)
# Ensure that the last index is included
# if reduced_tick_indices[-1] != len(HORIZONT)-1:
#   reduced_tick_indices = np.append(reduced_tick_indices, len(HORIZONT)-1)


# Set ticks and tick labels with correct values
ax.set_xticks(HORIZONT[reduced_tick_indices]) # Display ticks every 0.4 unit
ax.set_yticks(VERTICAL[reduced_tick_indices])
ax.set_xticklabels([f"{x:.1f}" for x in HORIZONT[reduced_tick_indices]], fontsize=12)  # Format x-axis labels
ax.set_yticklabels([f"{y:.1f}" for y in VERTICAL[reduced_tick_indices]], fontsize=12)  # Format y-axis labels


# Add a colorbar
cbar = plt.colorbar(im)
cbar.set_label('Predicted Species (0: Setosa, 1: Versicolor, 2: Virginica)', rotation=270, labelpad=25)

# Set the title
sepallen = 5.0
sepalwid = 3.0
ax.set_title(f"Species Classification with Sepal Length: {sepallen}, Sepal Width: {sepalwid}", fontsize=16)

plt.show()

print("Remember our species-to-number mapping:")
print("0 - setosa")
print("1 - versicolor")
print("2 - virginica")


#
# let's hold the petal length and width constant and vary sepal len + wid:

petallen = 3.42
petalwid = 3.42

VERTICAL = np.arange(0,10,.1) # array of vertical input values
HORIZONT = np.arange(0,10,.1) # array of horizontal input values
PLANEv2 = np.zeros( (len(HORIZONT),len(VERTICAL)) ) # the output array

row = 0
col = 0
for sepallen in VERTICAL: # for every sepal length
  for sepalwid in HORIZONT: # for every sepal width
    Features = [ sepallen, sepalwid, petallen, petalwid ]
    output = predictive_model(Features,knn_model_final)
    #print(f"Input {Features} Output: {output}")
    PLANEv2[row,col] = output
    col += 1
  col = 0
  row += 1
  print(".", end="")  # so we know it's running
  if row % 42 == 0: print() # same...

print("\n", PLANEv2[0:3,0:3]) # small bit of the upper-left corner



# prompt: please plot the above heatmap, with 1/4 as many axis labels

# Assuming 'PLANE', 'VERTICAL', and 'HORIZONT' are defined as in the original code
import matplotlib.pyplot as plt
# Create a new figure and axes
fig, ax = plt.subplots(figsize=(12, 8))

# Create the heatmap
im = ax.imshow(PLANEv2, cmap="viridis", extent=[HORIZONT.min(), HORIZONT.max(), VERTICAL.min(), VERTICAL.max()], origin="lower", aspect="auto")

# Set axis labels and ticks
ax.set_xlabel("sepalwid", fontsize=14)
ax.set_ylabel("sepallen", fontsize=14)

# Calculate the indices for reduced ticks and labels
reduced_tick_indices = np.arange(0, len(HORIZONT), len(HORIZONT)//8)
# Ensure that the last index is included
# if reduced_tick_indices[-1] != len(HORIZONT)-1:
#   reduced_tick_indices = np.append(reduced_tick_indices, len(HORIZONT)-1)


# Set ticks and tick labels with correct values
ax.set_xticks(HORIZONT[reduced_tick_indices]) # Display ticks every 0.4 unit
ax.set_yticks(VERTICAL[reduced_tick_indices])
ax.set_xticklabels([f"{x:.1f}" for x in HORIZONT[reduced_tick_indices]], fontsize=12)  # Format x-axis labels
ax.set_yticklabels([f"{y:.1f}" for y in VERTICAL[reduced_tick_indices]], fontsize=12)  # Format y-axis labels


# Add a colorbar
cbar = plt.colorbar(im)
cbar.set_label('Predicted Species (0: Setosa, 1: Versicolor, 2: Virginica)', rotation=270, labelpad=25)

# Set the title
sepallen = 5.0
sepalwid = 3.0
ax.set_title(f"Species Classification with petal Length: {petallen}, petal Width: {petalwid}", fontsize=16)

plt.show()

print("Remember our species-to-number mapping:")
print("0 - setosa")
print("1 - versicolor")
print("2 - virginica")


#
# That's it!  Welcome to the world of model-building workflows!!
#
#             Our prediction?  We'll be back for more ML!
#
# In fact, the rest of the hw is to run more ML workflows:   Births, Digits, Titanic, (ec) Housing, ...
#


#
# a coding cell placeholder
#

# You'll copy lots of cells - mostly coding cells - from the iris example




import pandas as pd

# Load dataset
births = pd.read_csv('births.csv')
births.head()


# Drop unnecessary columns
births = births.drop(columns=['Unnamed: 0', 'url'], errors='ignore')

# Rename the target column for clarity
births = births.rename(columns={'above/below median': 'popularity'})

# Convert "popularity" to numeric values: above → 1, below → 0
births['popularity'] = births['popularity'].map({'above': 1, 'below': 0})

# Drop rows where popularity mapping failed
births = births.dropna(subset=['popularity'])

# Define a function to filter valid calendar dates
def is_valid_date(month, day):
    try:
        pd.Timestamp(year=2020, month=int(month), day=int(day))
        return True
    except:
        return False

# Filter out rows with invalid dates (like Feb 30 or Apr 31)
births = births[births.apply(lambda row: is_valid_date(row['month'], row['day']), axis=1)]

# Make sure popularity is an integer
births['popularity'] = births['popularity'].astype(int)

births.head()



# Define features (X) and target (y)
X = births[['month', 'day']]
y = births['popularity'].astype(int)  # ensure integer type


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Try initial model with k = 5
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Initial Accuracy (k=5):", accuracy_score(y_test, y_pred))


from sklearn.model_selection import cross_val_score
import numpy as np

k_range = range(1, 21)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10)
    cv_scores.append(scores.mean())

# Best k
best_k = k_range[np.argmax(cv_scores)]
print("Best k:", best_k)



best_model = KNeighborsClassifier(n_neighbors=best_k)
best_model.fit(X_train, y_train)
final_pred = best_model.predict(X_test)

print("Final Accuracy (best_k):", accuracy_score(y_test, final_pred))


import seaborn as sns
import matplotlib.pyplot as plt

# Create a mesh grid of all valid month/day combinations
grid = pd.DataFrame([(m, d) for m in range(1, 13) for d in range(1, 32)], columns=['month', 'day'])
grid = grid[grid.apply(lambda row: is_valid_date(row['month'], row['day']), axis=1)]

# Predict popularity using the trained model
grid['predicted'] = best_model.predict(grid[['month', 'day']])

# Pivot for heatmap
heatmap_data = grid.pivot(index='day', columns='month', values='predicted')

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap='coolwarm', cbar_kws={'label': 'Predicted Popularity'})
plt.title('Predicted Birthday Popularity by Day and Month')
plt.xlabel('Month')
plt.ylabel('Day')
plt.show()



import pandas as pd

# Load the digits dataset
digits = pd.read_csv("digits.csv")

# View the first few rows
digits.head()



# Drop the column with the URL (if it exists)
digits = digits.drop(columns=['from http://chmullig.com/2012/06/births-by-day-of-year/'], errors='ignore')

# Confirm pixel and label structure
features = [f'pix{i}' for i in range(64)]
X = digits[features]  # 64 pixel columns
y = digits['actual_digit']  # digit label (0-9)

X.head(), y.head()



from sklearn.model_selection import train_test_split

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Try initial model with k = 3
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Initial accuracy (k=3):", accuracy_score(y_test, y_pred))



from sklearn.model_selection import cross_val_score
import numpy as np

k_range = range(1, 21)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10)
    cv_scores.append(scores.mean())

# Best k
best_k = k_range[np.argmax(cv_scores)]
print("Best k:", best_k)



# Train final model with best_k
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)

# Evaluate
final_pred = final_model.predict(X_test)
print("Final accuracy (best_k):", accuracy_score(y_test, final_pred))



import matplotlib.pyplot as plt
import numpy as np

# Convert first digit row to 8x8 array
first_digit_pixels = X.iloc[0].values.reshape(8, 8)
label = y.iloc[0]

# Plot
plt.figure(figsize=(4, 4))
plt.imshow(first_digit_pixels, cmap='gray')
plt.title(f"Digit: {label}", fontsize=16)
plt.axis('off')
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Store accuracies
accuracies = []

# Loop from 1 to 63 pixels (inclusive)
for P in range(1, 64):
    # Use the first P pixels
    pixel_cols = [f'pix{i}' for i in range(P)]
    X_partial = digits[pixel_cols]
    y = digits['actual_digit']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_partial, y, test_size=0.2, random_state=42)

    # Train model (we’ll use k=3 for simplicity)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Plot accuracy vs number of pixels used
plt.figure(figsize=(10, 6))
plt.plot(range(1, 64), accuracies, marker='o', color='blue')
plt.xlabel("Number of Pixels Used (P)")
plt.ylabel("Accuracy")
plt.title("Digit Classification Accuracy vs Number of Pixels Used")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()



