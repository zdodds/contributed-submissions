#
# Modern modeling ~ iris_modeler:  All-in-one iris clasification via nearest neighbors
#


# Section 1:  Libraries
#
import sklearn          # if not present, use a variant of  #3 install -U scikit-learn
import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)

# Section 2:  Read the already-cleaned data  (+ view, if you wish)
#
cleaned_filename = "iris_cleaned.csv"     # data should be "tidy" already...
df_tidy = pd.read_csv(cleaned_filename)   # can add encoding="utf-8" if needed
if False:
    print(f"{cleaned_filename} : file read into a pandas dataframe.")
    print("df_tidy is\n", df_tidy)
    print("df_tidy.info() is"); df_tidy.info()

# Section 3:  Drop any columns we don't want to use
ROW = 0
COLUMN = 1
df_model1 = df_tidy.drop('irisname', axis=COLUMN )
if False:  print("df_model1 is\n", df_model1)

# Section 4:  create COLUMNS and SPECIES variables to show we're organized + know what's happening...
COLUMNS = df_model1.columns                     # int to str
SPECIES = ['setosa','versicolor','virginica']   # int to str
SPECIES_INDEX = { s:i for i,s in enumerate(SPECIES) }  # str to int   {'setosa':0,'versicolor':1,'virginica':2}
COLUMNS_INDEX = { c:i for i,c in enumerate(COLUMNS) }  # str to int   {'sepallen':0,'sepalwid':1,'petallen':2, <more> }
if False:  print(f"{COLUMNS = } \n {COLUMNS_INDEX = } \n {SPECIES = } \n {SPECIES_INDEX = }")

# Section 5:  convert from pandas (spreadsheet) to numpy (array)
A = df_model1.to_numpy()    # yields the underlying numpy array
A = A.astype('float64')     # make sure everything is floating-point
NUM_ROWS, NUM_COLS = A.shape   # let's have NUM_ROWS and NUM_COLS around
if False:  print(f"\nThe dataset has {NUM_ROWS} rows and {NUM_COLS} cols")

# Section 6:  define our features (X_all) and our target-to-predict (y_all)
X_all = A[:,0:4]  # X (features) WATCH OUT! This is likely to change from model to model...
y_all = A[:,4]    # y (labels) WATCH OUT! This is likely to change from model to model...
if False:
    print(f"The labels/species are \n {y_all} \n ");
    print(f"The first few data rows are \n {X_all[0:5,:]}")

# Section 7:  80/20 split into training and testing sets:  X_train and y_train, X_test and y_test
from sklearn.model_selection import train_test_split      # this function splits into training + testing sets
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20)  # random_state=42 # 20% testing
if False:
    print(f"training with {len(y_train)} rows;  testing with {len(y_test)} rows\n" )
    print(f"Held-out testing data... (testing data: {len(y_test)} rows)")
    print(f"y_test: {y_test}")
    print(f"X_test (first few rows): {X_test[0:5,:]}\n")
    print(f"Training Data used for modeling... (training data: {len(y_train)} rows)")
    print(f"y_train: {y_train}")
    print(f"X_train (first few rows): {X_train[0:5,:]}")  # 5 rows

# Section 8:  Here's where the model-building happens!  First, we guess at the parameters (k=84)
from sklearn.neighbors import KNeighborsClassifier
k = 84   # we don't know what k to use, so we guess!  (this will _not_ be a good value)
knn_model = KNeighborsClassifier(n_neighbors=k)       # here, k is the "k" in kNN
knn_model.fit(X_train, y_train)      # we train the model ... it's one line!
if False:  print("Created and trained a knn classifier with k =", k)

# Section 9:  Let's see how our naive model does on the TEST data!
predicted_labels = knn_model.predict(X_test)      # THIS IS THE KEY LINE:  predict
actual_labels = y_test
if True:
    print("Predicted labels:", predicted_labels)
    print("Actual  labels  :", actual_labels)
    num_correct = sum(predicted_labels == actual_labels)
    total = len(actual_labels)
    print(f"\nResults on test set:  {num_correct} correct out of {total} total, for {num_correct*100/total}%\n")

# Section 10:  Let's cross-validate to find the "best" value of k, best_k:
import time
from sklearn.model_selection import cross_val_score
all_accuracies = []
best_k = 84  # Not correct!
best_accuracy = 0.0  # also not correct...
for k in range(1,85):    # Note that we are cross-validating using only our TRAINING data!
    knn_cv_model = KNeighborsClassifier(n_neighbors=k)   # build a knn_model for every k
    cv_scores = cross_val_score( knn_cv_model, X_train, y_train, cv=5 )  # cv=5 means 80/20
    this_cv_accuracy = cv_scores.mean()               # mean() is numpy's built-in average function
    if True: print(f"k: {k:2d}  cv accuracy: {this_cv_accuracy:7.4f}")
    if this_cv_accuracy > best_accuracy:  # is this one better?
        best_accuracy = this_cv_accuracy  # track the best accuracy
        best_k = k                        # with the best k
    all_accuracies.append(this_cv_accuracy)
    time.sleep(0.002)   # dramatic pauses!
if True: print(f"best_k = {best_k}  \n    yields the highest cv accuracy: {best_accuracy}")  # print the best one




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
# Ok!  We have our knn model, let's use it...
#
# ... in a data-trained predictive model (k-nearest-neighbors), using scikit-learn
#
# warning: this model has NOT yet been tuned to its "best k"
#
def predictive_model( Features ):
    """ input: a list of four features
                [ sepallen, sepalwid, petallen, petalwid ]
        output: the predicted species of iris, from
                  setosa (0), versicolor (1), virginica (2)
    """
    our_features = np.asarray([Features])                      # extra brackets needed so it's 2d
    predicted_species_list = knn_model.predict(our_features)   # PREDICT!

    predicted_species = int(round(predicted_species_list[0]))  # unpack the one element it contains
    name = SPECIES[predicted_species]                          # look up the species
    return name

#
# Try it!
#
# Features = eval(input("Enter new Features: "))
#
ListofFeatures = [ [6.7,3.3,5.7,2.1],
                   [5.8,2.7,4.1,1.0],
                   [4.6,3.6,3.0,2.2],
                   [6.7,3.3,5.7,2.1],
                   [4.2,4.2,4.2,4.2],
                   [1,42,4.7,3.01],        # -4.7? .01?  0?
                   ]

for Features in ListofFeatures:
    result = predictive_model( Features )
    print(f"From the Features {Features}, I predict {result}")


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

def predictive_model( Features, Model ):                 # to allow the input of any Model
    """ input: a list of four features
                [ sepallen, sepalwid, petallen, petalwid ]
        output: the predicted species of iris, from
                  setosa (0), versicolor (1), virginica (2)
    """
    our_features = np.asarray([Features])                 # extra brackets needed for 2d
    predicted_species = Model.predict(our_features)       # The model's prediction!
    predicted_species = int(round(predicted_species[0]))  # unpack the extra brackets
    return predicted_species

#
# Try it!
#

LoF = [
[4.8, 3.1, 1.6, 0.2 ],   # actually setosa
[5.7, 2.9, 4.2, 1.3 ],   # actually versicolor
[5.8, 2.7, 5.1, 1.9 ],   # actually virginica
[5.2, 4.1, 1.5, 0.1 ],   # actually setosa
[5.4, 3.4, 1.5, 0.4 ],   # actually setosa
[5.1, 2.5, 3.0, 1.1 ],   # actually versicolor
[6.2, 2.9, 4.3, 1.3 ],   # actually versicolor
[6.3, 3.3, 6.0, 2.5 ],   # actually virginica
[5.7, 2.8, 4.1, 1.3 ],   # actually virginica  <-- almost always wrong!

[0,0,0,0],               # used as a separator here

[3.7, 2.8, 2.1, 0.3 ],   # let's use this for our own "new" iris ...
]

# run on each one:
for Features in LoF:
    predicted_species = predictive_model( Features, knn_model_final )  # pass in the model, too!
    name = SPECIES[predicted_species]
    print(f"I predict {name} from the features {Features}")    # Answers in the assignment...


#
# That's it!  Welcome to the world of model-building workflows!!
#
#             Our prediction?  We'll be back for more ML!
#
# In fact, the rest of the hw is to run more ML workflows:
# Births, Digits, another dataset, which could be Titanic, Housing, ...
#
# and more ML algorithms:
# Decision Trees, Random Forests, Neural Nets
# and, optionally, time series, recommendation systems, ...


# we can only plot 2 dimensions at a time!
# These two will be our constants:
sepallen = 5.0
sepalwid = 3.0
# petallen =
# petalwid =

VERTICAL = np.arange(0,8,.1) # array of vertical input values
HORIZONT = np.arange(0,8,.1) # array of horizontal input values
PLANE = np.zeros( (len(HORIZONT),len(VERTICAL)) ) # the output array
MODEL = knn_model_final


col = 0
row = 0
for petallen in VERTICAL: # for every sepal length
  for petalwid in HORIZONT: # for every sepal width
    Features = [ sepallen, sepalwid, petallen, petalwid ]
    output = predictive_model(Features,MODEL)
    #print(f"Input {Features} Output: {output}")
    PLANE[row,col] = output
    row += 1
  row = 0
  col += 1
  print(".", end="")  # so we know it's running
  if col % 42 == 0: print() # same...

print("\n", PLANE[0:3,0:3]) # small bit of the lower-left corner


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


import numpy as np
import pandas as pd

filename='births.csv'
df=pd.read_csv(filename)
print(f"{filename} : file read into a pandas dataframe.")





df.info()


for column_name in df.columns:
    print(f"{column_name =}")


ROW = 0
COLUMN = 1

df_clean1 = df.drop(['births','from http://chmullig.com/2012/06/births-by-day-of-year/'], axis=COLUMN)
df_clean1


df_clean2 = df_clean1.dropna()
df_clean2.info()

df_clean2


df_clean2.info()


COLUMNS = df_clean1.columns
print(f"COLUMNS is {COLUMNS}")

print(f"COLUMNS[0] is {COLUMNS[0]}\n")

COL_INDEX = {}
for i, name in enumerate(COLUMNS):
    COL_INDEX[name] = i
print(f"COL_INDEX is {COL_INDEX}")


MEDIAN = ['below','above']
MEDIAN_INDEX = {'above':1,'below':0}

def convert_median(median):

    return MEDIAN_INDEX[median]

for median in MEDIAN:
    print(f"{median} maps to {convert_median(median)}")


convert_median( 'below')


df_clean3 = df_clean2.copy()

df_clean3['above/below median (num)'] = df_clean2['above/below median'].apply(convert_median)

df_clean3


df_tidy =  df_clean3
old_basename = filename[:-4]
cleaned_filename = old_basename + "_cleaned.csv"
print(f"cleaned_filename is {cleaned_filename}")

df_tidy.to_csv(cleaned_filename, index_label=False)


df_tidy_reread = pd.read_csv(cleaned_filename)
print(f"{filename} : file read into a pandas dataframe.")
df_tidy_reread


COLUMNS = df_tidy.columns
print(f"COLUMNS is {COLUMNS}\n")

print(f"COLUMNS[0] is {COLUMNS[0]}\n")

COL_INDEX = {}
for i, name in enumerate(COLUMNS):
    COL_INDEX[name] = i
print(f"COL_INDEX is {COL_INDEX}\n\n")

MEDIAN = ['below','above']
MEDIAN_INDEX = {'below':0,'above':1}

def convert_median(median):
    """ return the species index (a unique integer/category) """

    return MEDIAN_INDEX[median]

for median in MEDIAN:
    print(f"{median} maps to {convert_median(median)}")



import numpy as np
import pandas as pd

cleaned_filename = "births_cleaned.csv"
df_tidy = pd.read_csv(cleaned_filename)
print(f"{cleaned_filename} : file read into a pandas dataframe.")
df_tidy


pd.set_option('display.max_rows',20)
pd.set_option('max_colwidth', 400)
df_tidy


df_tidy.info()


ROW = 0
COLUMN = 1
df_model1 = df_tidy.drop( 'above/below median', axis=COLUMN )
df_model1


COLUMNS = df_model1.columns
print(f"COLUMNS is {COLUMNS}\n")
print(f"COLUMNS[0] is {COLUMNS[0]}\n")

COL_INDEX = {}
for i, name in enumerate(COLUMNS):
    COL_INDEX[name] = i
print(f"COL_INDEX is {COL_INDEX}\n\n")

MEDIAN = ['below','above']
MEDIAN_INDEX = {'below':0,'above':1}

def convert_median(median):

    return MEDIAN_INDEX[median]

for median in MEDIAN:
    print(f"{median} maps to {convert_median(median)}")


A = df_model1.to_numpy()
print(A)


A = A.astype('int64')
print(A[0:5])


NUM_ROWS, NUM_COLS = A.shape
print(f"\nThe dataset has {NUM_ROWS} rows and {NUM_COLS} cols")


df_tidy.info()


n = 35
print(f"births #{n} is {A[n]}")

for i in range(len(COLUMNS)):
    colname = COLUMNS[i]
    value = A[n][i]
    print(f"  The {colname} is {value}")

median_index = COL_INDEX['above/below median (num)']
median_num = int(round(A[n][median_index]))
median = MEDIAN[median_num]
print(f"  The median is {median} (i.e., {median_num})")


print("+++ Start of data definitions +++\n")

X_all = A[:,0:2]
y_all = A[:,2]

print(f"y_all (just the labels)   are \n {y_all}")
print(f"X_all (just the features - a few) are \n {X_all[0:5]}")


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2)

print(f"training with {len(y_train)} rows;  testing with {len(y_test)} rows\n" )

print("+++++")
print(f"Held-out data... (testing data: {len(y_test)} rows)")
print("+++++\n")
print(f"y_test: {y_test}")
print(f"X_test (first few rows): {X_test[0:5,:]}")
print()



print("+++++")
print(f"Data used for modeling... (training data: {len(y_train)} rows)")
print("+++++\n")
print(f"y_train: {y_train}")
print(f"X_train (first few rows): {X_train[0:5,:]}")


from sklearn.neighbors import KNeighborsClassifier

k = 84
knn_model = KNeighborsClassifier(n_neighbors=k)

knn_model.fit(X_train, y_train)
print("Created and trained a knn classifier with k =", k)


predicted_labels = knn_model.predict(X_test)
actual_labels = y_test
print("Predicted labels:", predicted_labels)
print("Actual  labels  :", actual_labels)

num_correct = sum(predicted_labels == actual_labels)
total = len(actual_labels)
print(f"\nResults on test set:  {num_correct} correct out of {total} total.")


def compare_labels(predicted_labels, actual_labels):
    """ a more neatly formatted comparison """
    NUM_LABELS = len(predicted_labels)
    num_correct = 0

    print()
    print(f'row {"#":>3s} : {"predicted":>12s} {"actual":<12s}   {"result"}')

    for i in range(NUM_LABELS):
        p = int(round(predicted_labels[i]))
        a = int(round(actual_labels[i]))
        result = "incorrect"
        if p == a:
            result = ""
            num_correct += 1

        print(f"row {i:>3d} : {MEDIAN[p]:>12s} {MEDIAN[a]:<12s}   {result}")

    print()
    print("Correct:", num_correct, "out of", NUM_LABELS)
    return num_correct
compare_labels(predicted_labels,actual_labels)


def predictive_model( Features ):

    our_features = np.asarray([Features[:2]])
    predicted_median = knn_model.predict(our_features)

    predicted_median = int(round(predicted_median[0]))
    name = MEDIAN[predicted_median]
    return name

Features = [-7006.7,3900.3,0.7,80]
result = predictive_model( Features )
print(f"I predict {result} from Features {Features}")


from sklearn.model_selection import cross_val_score
best_k = 84
best_accuracy = 0

for k in range(1,85):
    knn_cv_model = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score( knn_cv_model, X_train, y_train, cv=5 )
    this_cv_accuracy = cv_scores.mean()
    print(f"k: {k:2d}  cv accuracy: {this_cv_accuracy:7.4f}")

    if this_cv_accuracy > best_accuracy:
        best_accuracy = this_cv_accuracy
        best_k = k

print(f"best_k = {best_k}   yields the highest average cv accuracy.")  # print the best one



from sklearn.neighbors import KNeighborsClassifier
knn_model_tuned = KNeighborsClassifier(n_neighbors=best_k)

knn_model_tuned.fit(X_train, y_train)
print(f"Created + trained a knn classifier, now tuned with a (best) k of {best_k}")



predicted_labels = knn_model_tuned.predict(X_test)
actual_labels = y_test

print("Predicted labels:", predicted_labels)
print("Actual labels:", actual_labels)

num_correct = sum(predicted_labels == actual_labels)
total = len(actual_labels)
print(f"\nResults on test set:  {num_correct} correct out of {total} total.\n\n")

compare_labels(predicted_labels,actual_labels)


knn_model_final = KNeighborsClassifier(n_neighbors=best_k)
knn_model_final.fit(X_all, y_all)
print(f"Created + trained a 'final' knn classifier, with a (best) k of {best_k}")


def predictive_model( Features, Model ):
    our_features = np.asarray([Features[:2]])
    predicted_median = Model.predict(our_features)
    predicted_median = int(round(predicted_median[0]))
    return predicted_median

LoF = [
[1, 1 ],
[2, 27 ],
[3, 7 ],
[4, 10],
[5, 5 ],
[6, 5 ],
[10, 18 ],
[11, 6 ],
[12, 31 ],
]

for Features in LoF:
    predicted_median = predictive_model( Features, knn_model_final )
    name = MEDIAN[predicted_median]
    print(f"I predict {name} median from the features {Features}")


day = 11
month = 1

VERTICAL = np.arange(0,10,.1)
HORIZONT = np.arange(0,10,.1)
PLANE = np.zeros( (len(HORIZONT),len(VERTICAL)) )

row = 0
col = 0
for day in VERTICAL:
  for month in HORIZONT:
    Features = [ day,month]
    output = predictive_model(Features,knn_model_final)
    PLANE[row,col] = output
    col += 1
  col = 0
  row += 1
  print(".", end="")
  if row % 42 == 0: print()

print("\n", PLANE[0:3,0:3])


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


import seaborn as sns
sns.set(rc = {'figure.figsize':(12,8)})
ax = sns.heatmap(PLANE)
ax.invert_yaxis()
ax.set(xlabel="month", ylabel="day")
ax.set_xticks(ax.get_xticks()[::4])
ax.set_yticks(ax.get_yticks()[::4])

print("Remember our below/above median-to-number mapping:")
print("0 - below")
print("1- above")


month = 5

VERTICAL = np.arange(0,10,.1)
HORIZONT = np.arange(0,10,.1)
PLANEv2 = np.zeros( (len(HORIZONT),len(VERTICAL)) )

row = 0
col = 0
for day in VERTICAL:
  for month in HORIZONT:
    Features = [ day,month ]
    output = predictive_model(Features,knn_model_final)
    PLANEv2[row,col] = output
    col += 1
  col = 0
  row += 1
  print(".", end="")
  if row % 42 == 0: print()

print("\n", PLANEv2[0:3,0:3])


import seaborn as sns

sns.set(rc = {'figure.figsize':(12,8)})
ax = sns.heatmap(PLANEv2)
ax.invert_yaxis()
ax.set(xlabel="month (tenths)", ylabel="day (tenths)")
ax.set_xticks(ax.get_xticks()[::4])
ax.set_yticks(ax.get_yticks()[::4])

print("Remember our below/above median-to-number mapping:")
print("0 - below")
print("1- above")


