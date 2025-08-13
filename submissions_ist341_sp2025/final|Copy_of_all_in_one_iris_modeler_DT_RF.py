#
# Modern modeling ~ iris_modeler:  All-in-one iris clasification via DT + RF
#


# Section 1:  Libraries
#
import time
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

# Section 8:  Here's where the model-building happens!  First, we guess at the parameters
from sklearn import tree      # for decision trees
best_depth = 1   # we don't know what depth to use, so let's guess 1 (not a good guess)
dtree_model = tree.DecisionTreeClassifier(max_depth=best_depth)
dtree_model.fit(X_train, y_train)      # we train the model ... it's one line!
if False:  print("Created and trained a classifier with best_depth =", best_depth)

# Section 9:  Let's see how our naive model does on the TEST data!
predicted_labels = dtree_model.predict(X_test)      # THIS IS THE KEY LINE:  predict
actual_labels = y_test
if False:
    print("Predicted labels:", predicted_labels)
    print("Actual  labels  :", actual_labels)
    num_correct = sum(predicted_labels == actual_labels)
    total = len(actual_labels)
    print(f"\nResults on test set:  {num_correct} correct out of {total} total, for {num_correct*100/total}%\n")

# Section 10:  Let's cross-validate to find the "best" value of k, best_k:
print("Cross-validating...")
from sklearn.model_selection import cross_val_score
all_accuracies = []
best_depth = 1   # we don't know what depth to use, so let's guess 1 (not a good guess)
best_accuracy = 0.0  # also not correct...
for depth in range(1,6):    # Note that we are cross-validating using only our TRAINING data!
    dtree_cv_model = tree.DecisionTreeClassifier(max_depth=depth)   # build a knn_model for every k
    cv_scores = cross_val_score( dtree_cv_model, X_train, y_train, cv=5 )  # cv=5 means 80/20
    this_cv_accuracy = cv_scores.mean()               # mean() is numpy's built-in average function
    if False: print(f"depth: {depth:2d}  cv accuracy: {this_cv_accuracy:7.4f}")
    if this_cv_accuracy > best_accuracy:  # is this one better?
        best_accuracy = this_cv_accuracy  # track the best accuracy
        best_depth = depth                        # with the best k
    all_accuracies.append(this_cv_accuracy)
    time.sleep(0.002)   # dramatic pauses!
if True: print(f"best_depth = {best_depth}  \n    yields the highest cv accuracy: {best_accuracy}\n")  # print the best one

# Section 11:  Here's where the model-building happens with the best-found parameters:
dtree_model_final = tree.DecisionTreeClassifier(max_depth=best_depth)
dtree_model_final.fit(X_all, y_all)      # we train the model ... on _all_ the data!
if True:  print("Created and trained a classifier with best_depth =", best_depth)
# print the feature importances...
if False:  print("\nThe feature importances are", dtree_model_final.feature_importances_)


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

# let's try it out!  use the model you want:
predicted_labels = dtree_model_final.predict(X_test)      # THIS IS THE KEY LINE:  predict
compare_labels(predicted_labels,actual_labels)


import matplotlib.pyplot as plt

FEATURES = COLUMNS[0:4]

#
# Now, let's see the tree!
#

filename = 'tree_data.gv'    # sometimes .dot is used, instead of .gv
model = dtree_model_final

tree.export_graphviz(model, out_file=filename,  # the filename constructed above...!
                            feature_names=COLUMNS[:-1], # actual feature names, not species
                            filled=True,              # fun!
                            rotate=False,             # False for Up/Down; True for L/R
                            class_names=SPECIES,      # good to have
                            leaves_parallel=True )    # lots of options!

print(f"file {filename} written. Try pasting its contents to  http://viz-js.com/\n")

with open(filename, "r") as f:
    all_file_text = f.read()
    print(all_file_text)

#
# Tree display...
#
fig = plt.figure(figsize=(12,8))
tree_plot = tree.plot_tree(model,
                   feature_names=FEATURES,   # Glad to have these!
                   class_names=SPECIES,      # and these!!
                   filled=True)

plt.show()


#
# Ok!  We have our model, let's use it...
#
# ... in a data-trained predictive model (k-nearest-neighbors), using scikit-learn
#
# warning: this model has NOT yet been tuned to its "best k"
#
def predictive_model( Features, model ):
    """ input: a list of four features
                [ sepallen, sepalwid, petallen, petalwid ]
        output: the predicted species of iris, from
                  setosa (0), versicolor (1), virginica (2)
    """
    our_features = np.asarray([Features])                      # extra brackets needed so it's 2d
    predicted_species_list = model.predict(our_features)   # PREDICT!

    predicted_species = int(round(predicted_species_list[0]))  # unpack the one element it contains
    return predicted_species

#
# Try it!
#
# Features = eval(input("Enter new Features: "))
#
ListofFeatures = [ [4.2,3.1,2.0,0.4],
                   [5.8,2.7,4.1,1.0],
                   [4.6,3.6,3.0,2.2],
                   [6.7,3.3,5.7,2.1],
                   [4.2,4.2,4.2,4.2],
                   [1.0,42,4.7,0.01],        # -4.7? .01?  0?
                   ]

for Features in ListofFeatures:
    predicted_species = predictive_model( Features, dtree_model_final )
    name = SPECIES[predicted_species]                          # look up the species
    print(f"From the Features {Features}, I predict : {name}")


# we can only plot 2 dimensions at a time!
# These two will be our constants:
sepallen = 4.0
sepalwid = 2.0
# petallen =
# petalwid =

VERTICAL = np.arange(0,8,.1) # array of vertical input values
HORIZONT = np.arange(0,8,.1) # array of horizontal input values
PLANE = np.zeros( (len(HORIZONT),len(VERTICAL)) ) # the output array
MODEL = dtree_model_final


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


import seaborn as sns
# sns.heatmap(PLANE)

sns.set(rc = {'figure.figsize':(12,8)})  # figure size!
ax = sns.heatmap(PLANE)
ax.invert_yaxis() # to match our usual direction
ax.set(xlabel="petalwid (tenths)", ylabel="petallen (tenths)")
ax.set_xticks(ax.get_xticks()[::4])
ax.set_yticks(ax.get_yticks()[::4])
ax.set_title("DT: Prediction Landscape for sepallen = 5.0 and sepalwid = 3.0", fontsize=18)


print("Remember our species-to-number mapping:")
print("0 - setosa")
print("1 - versicolor")
print("2 - virginica")


# Section 1:  Libraries
#
import time
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

# Section 8:  Here's where the model-building happens!  First, we guess at the parameters
from sklearn import tree      # for decision trees
from sklearn import ensemble  # for random forests, an ensemble classifier
best_d = 1            # we don't know what depth to use, so let's guess 1 (not a good guess)
best_num_trees = 42   # we don't know how many trees to use, so let's guess 42
rforest_model = ensemble.RandomForestClassifier(max_depth=best_d, n_estimators=best_num_trees, max_samples=0.5)  # 0.5 of the data each tree
rforest_model.fit(X_train, y_train)      # we train the model ... it's one line!
if False:  print(f"Built a Random Forest with depth={best_d} and number of trees={best_num_trees}")

# Section 9:  Let's see how our naive model does on the TEST data!
predicted_labels = rforest_model.predict(X_test)      # THIS IS THE KEY LINE:  predict
actual_labels = y_test
if False:
    print("Predicted labels:", predicted_labels)
    print("Actual  labels  :", actual_labels)
    num_correct = sum(predicted_labels == actual_labels)
    total = len(actual_labels)
    print(f"\nResults on test set:  {num_correct} correct out of {total} total, for {num_correct*100/total}%\n")

# Section 10:  Let's cross-validate to find the "best" value of k, best_k:
print("Cross-validating...")
from sklearn.model_selection import cross_val_score
all_accuracies = []
best_d = 1         # range(1,6)
best_num_trees = 50   # [50,150,250]
best_accuracy = 0
for d in range(1,6):
    for ntrees in [50,150,250]:
        rforest_model = ensemble.RandomForestClassifier(max_depth=d, n_estimators=ntrees,max_samples=0.5)
        cv_scores = cross_val_score( rforest_model, X_train, y_train, cv=5 ) # 5 means 80/20 split
        average_cv_accuracy = cv_scores.mean()  # more likely, only their average
        if True: print(f"depth: {d:2d} ntrees: {ntrees:3d} cv accuracy: {average_cv_accuracy:7.4f}")
        if average_cv_accuracy > best_accuracy:
            best_accuracy = average_cv_accuracy;   best_d = d;      best_num_trees = ntrees
if True: print(f"best_depth: {best_depth} and best_num_trees: {best_num_trees} are our choices. Acc: {best_accuracy}")

# Section 11:  Here's where the model-building happens with the best-found parameters:
rforest_model_tuned = ensemble.RandomForestClassifier(max_depth=best_depth, n_estimators=best_num_trees, max_samples=0.5)
rforest_model_tuned.fit(X_all, y_all)      # we train the model ... on _all_ the data!
if True:  print("Created and trained a classifier with best_depth =", best_depth)
# print the feature importances...
if False:  print("\nThe feature importances are", rforest_model_tuned.feature_importances_)


#
# we can get the individual trees, if we want...  Let's try it on tree #28
#
tree_index = 28   # which tree
one_rf_tree = rforest_model_tuned.estimators_[tree_index]
print(f"One of the forest's trees is {one_rf_tree}")

# From there, it's possible to create a graphical version...
filename = f'rf_tree_{tree_index:03d}.gv'             # f strings! Could save all trees, but we won't do so here.
tree.export_graphviz(one_rf_tree, out_file=filename,  # the filename constructed above...!
                            feature_names=FEATURES, # actual feature names, not species
                            filled=True,              # fun!
                            rotate=False,             # False for Up/Down; True for L/R
                            class_names=SPECIES,      # good to have
                            leaves_parallel=True )    # lots of options!

print(f"file {filename} written. Try copying the result to http://viz-js.com/ \n")
with open(filename, "r") as f:
    file_text = f.read()
    print(file_text)

# One tree:
fig = plt.figure(figsize=(10,8))
tree_plot = tree.plot_tree(one_rf_tree,
                   feature_names=FEATURES,   # Glad to have these!
                   class_names=SPECIES,      # and these!!
                   filled=True)


#
# final predictive model (random forests), with tuned parameters + ALL data incorporated
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
Features = [6.7,3.3,5.7,0.1]  # [5.8,2.7,4.1,1.0] [4.6,3.6,3.0,2.2] [6.7,3.3,5.7,2.1]

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
]

# run on each one:
for Features in LoF:
    predicted_species = predictive_model( Features, rforest_model_tuned )  # pass in the model, too!
    name = SPECIES[predicted_species]
    print(f"from the features {Features} I predict {name}")    # Answers in the assignment...


# we can only plot 2 dimensions at a time!
# These two will be our constants:
sepallen = 5.0
sepalwid = 3.0
# petallen =
# petalwid =

VERTICAL = np.arange(0,8,.1) # array of vertical input values
HORIZONT = np.arange(0,8,.1) # array of horizontal input values
PLANE = np.zeros( (len(HORIZONT),len(VERTICAL)) ) # the output array
MODEL = rforest_model_tuned

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

print("\n", PLANE[0:3,0:3]) # small bit of the upper-left corner


import seaborn as sns
# sns.heatmap(PLANE)

#sns.set(rc = {'figure.figsize':(18,12)})  # figure size!

fig, ax = plt.subplots(figsize=(12,8))

# Create the heatmap
im = ax.imshow(PLANE, cmap="viridis", extent=[HORIZONT.min(), HORIZONT.max(), VERTICAL.min(), VERTICAL.max()], origin="lower", aspect="auto")

# Set axis labels and ticks
ax.set_xlabel("petalwid", fontsize=14)
ax.set_ylabel("petallen", fontsize=14)

ax.set_title(f"RF: petallen vs petalwid with sepallen == {sepallen:.1f} and sepalwid == {sepalwid:.1f}\n", fontsize=18)
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

plt.show()

print("Remember our species-to-number mapping:")
print("0 - setosa")
print("1 - versicolor")
print("2 - virginica")


rforest_model_tuned.feature_importances_


# births

import time
import sklearn
import numpy as np
import pandas as pd

cleaned_filename = "births_cleaned.csv"
df_tidy = pd.read_csv(cleaned_filename)
df_model0 = df_tidy.drop('above/below median', axis=1)
df_model1 = df_model0.drop('births', axis=1)

# organizing... so fun... I'm very organized......
COLUMNS = df_model1.columns
POPULARITY = ['below','above']
POPULARITY_INDEX = { s:i for i,s in enumerate(POPULARITY) }
COLUMNS_INDEX = { c:i for i,c in enumerate(COLUMNS) }

# munpy! I mean-- numpy!!!
A = df_model1.to_numpy()
A = A.astype('float64')
NUM_ROWS, NUM_COLS = A.shape

# (features, what to predict)
X_all = A[:,0:2]
y_all = A[:,2]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20)
from sklearn import tree
best_depth = 3
dtree_model = tree.DecisionTreeClassifier(max_depth=best_depth)
dtree_model.fit(X_train, y_train)

# teeeeeeeeeeeeeeeest!!!!!!
predicted_labels = dtree_model.predict(X_test)
actual_labels = y_test
if False:
  print("Predicted labels:", predicted_labels)
  print("Actual  labels  :", actual_labels)
  num_correct = sum(predicted_labels == actual_labels)
  total = len(actual_labels)
  print(f"\nResults on test set:  {num_correct} correct out of {total} total, for {num_correct*100/total}%\n")

# cross multiply-- I mean validate! of course, haha...
from sklearn.model_selection import cross_val_score
all_accuracies = []
best_depth = 3
best_accuracy = 0.0
for depth in range(1,10):
  dtree_cv_model = tree.DecisionTreeClassifier(max_depth=depth)
  cv_scores = cross_val_score( dtree_cv_model, X_train, y_train, cv=5 )
  this_cv_accuracy = cv_scores.mean()
  if this_cv_accuracy > best_accuracy:
      best_accuracy = this_cv_accuracy
      best_depth = depth
  all_accuracies.append(this_cv_accuracy)
print(f"best_depth = {best_depth}  \n    yields the highest cv accuracy: {best_accuracy}\n")

# final!!! ...but the best depth keeps changing T v T
dtree_model_final = tree.DecisionTreeClassifier(max_depth=best_depth)
dtree_model_final.fit(X_all, y_all)
if True:  print("Created and trained a classifier with best_depth =", best_depth)
print("\nThe feature importances are", dtree_model_final.feature_importances_)



# T R E E

import matplotlib.pyplot as plt
FEATURES = COLUMNS[0:2]

filename = 'tree_data.gv'
model = dtree_model_final

tree.export_graphviz(model, out_file=filename,  # the filename constructed above...!
                            feature_names=COLUMNS[:-1], # actual feature names
                            filled=True,              # fun!
                            rotate=False,             # False for Up/Down; True for L/R
                            class_names=POPULARITY,      # good to have
                            leaves_parallel=True )    # lots of options!

print(f"file {filename} written. Try pasting its contents to  http://viz-js.com/\n")

with open(filename, "r") as f:
    all_file_text = f.read()
    print(all_file_text)

#
# Tree display...
#
fig = plt.figure(figsize=(12,8))
tree_plot = tree.plot_tree(model,
                   feature_names=FEATURES,   # Glad to have these!
                   class_names=POPULARITY,      # and these!!
                   filled=True)

plt.show()


def predictive_model( Features, model ):
    """ input: a list of two features
                [month, day]
        output: the predicted popularity of births on that day:
                                          above or below median
    """
    our_features = np.asarray([Features])
    predicted_popularity_list = model.predict(our_features)

    predicted_popularity = int(round(predicted_popularity_list[0]))
    return predicted_popularity

ListofFeatures = [ [5,9],
                   [6, 18],
                   [2,25],
                   [20,11],
                   [1,10],
                   [1,34],
                   ]

for Features in ListofFeatures:
    predicted_popularity = predictive_model( Features, dtree_model_final )
    name = POPULARITY[predicted_popularity]
    print(f"From the Features {Features}, I predict : {name}")


VERTICAL = np.arange(0,12,1)
HORIZONT = np.arange(0,31,1)
PLANE = np.zeros( (len(HORIZONT),len(VERTICAL)) )
MODEL = dtree_model_final

col = 0
row = 0
for month in VERTICAL:
  for day in HORIZONT:
    Features = [month, day]
    output = predictive_model(Features,MODEL)
    PLANE[row,col] = output
    row += 1
  row = 0
  col += 1
  print(".", end="")
print("\n", PLANE[0:30,0:10])

import seaborn as sns

sns.set(rc = {'figure.figsize':(12,8)})
ax = sns.heatmap(PLANE)
ax.invert_yaxis()
ax.set(xlabel="month", ylabel="day")
ax.set_xticks(ax.get_xticks()[::2])
ax.set_yticks(ax.get_yticks()[::4])
ax.set_title("DT: Prediction for Popularity of Birthdays", fontsize=18)


# births
# forests are cool. I think.

import time
import sklearn
import numpy as np
import pandas as pd

cleaned_filename = "births_cleaned.csv"
df_tidy = pd.read_csv(cleaned_filename)
df_model0 = df_tidy.drop('above/below median', axis=1)
df_model1 = df_model0.drop('births', axis=1)

# organizing... so fun... I'm very organized......
COLUMNS = df_model1.columns
POPULARITY = ['below','above']
POPULARITY_INDEX = { s:i for i,s in enumerate(POPULARITY) }
COLUMNS_INDEX = { c:i for i,c in enumerate(COLUMNS) }

# munpy! I mean-- numpy!!!
A = df_model1.to_numpy()
A = A.astype('float64')
NUM_ROWS, NUM_COLS = A.shape

# (features, what to predict)
X_all = A[:,0:2]
y_all = A[:,2]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20)
from sklearn import tree
from sklearn import ensemble # Y E E E E L L L L L L L L L L L L below this changes in random forests ANKDDANKDNKDNNVKNDNK
best_d = 3
best_num_trees = 33
rforest_model = ensemble.RandomForestClassifier(max_depth=best_d, n_estimators=best_num_trees, max_samples=0.5)
rforest_model.fit(X_train, y_train)

# teeeeeeeeeeeeeeeest!!!!!!
predicted_labels = rforest_model.predict(X_test)
actual_labels = y_test
if False:
  print("Predicted labels:", predicted_labels)
  print("Actual  labels  :", actual_labels)
  num_correct = sum(predicted_labels == actual_labels)
  total = len(actual_labels)
  print(f"\nResults on test set:  {num_correct} correct out of {total} total, for {num_correct*100/total}%\n")

# cross multiply-- I mean validate! of course, haha...
from sklearn.model_selection import cross_val_score
all_accuracies = []
best_d = 3
best_num_trees = 50
best_accuracy = 0.0
for d in range(1,6):
  for ntrees in [50,100,150]:
    rforest_model = ensemble.RandomForestClassifier(max_depth=d, n_estimators=ntrees,max_samples=0.5)
    cv_scores = cross_val_score(rforest_model, X_train, y_train, cv=5)
    average_cv_accuracy = cv_scores.mean()
    if average_cv_accuracy > best_accuracy:
      best_accuracy = average_cv_accuracy;   best_d = d;      best_num_trees = ntrees
print(f"best_d: {best_d} and best_num_trees: {best_num_trees} are our choices. Acc: {best_accuracy}")

rforest_model_tuned = ensemble.RandomForestClassifier(max_depth=best_d, n_estimators=best_num_trees, max_samples=0.5)
rforest_model_tuned.fit(X_all, y_all)
print("\nThe feature importances are", rforest_model_tuned.feature_importances_)


#
# we can get the individual trees, if we want...  Let's try it on tree #28
#
tree_index = 28   # which tree
one_rf_tree = rforest_model_tuned.estimators_[tree_index]
print(f"One of the forest's trees is {one_rf_tree}")

# From there, it's possible to create a graphical version...
filename = f'rf_tree_{tree_index:03d}.gv'             # f strings! Could save all trees, but we won't do so here.
tree.export_graphviz(one_rf_tree, out_file=filename,  # the filename constructed above...!
                            feature_names=FEATURES, # actual feature names, not species
                            filled=True,              # fun!
                            rotate=False,             # False for Up/Down; True for L/R
                            class_names=POPULARITY,      # good to have
                            leaves_parallel=True )    # lots of options!

print(f"file {filename} written. Try copying the result to http://viz-js.com/ \n")
with open(filename, "r") as f:
    file_text = f.read()
    print(file_text)

# One tree:
fig = plt.figure(figsize=(10,8))
tree_plot = tree.plot_tree(one_rf_tree,
                   feature_names=FEATURES,   # Glad to have these!
                   class_names=POPULARITY,      # and these!!
                   filled=True)


# it takes up so much s p a c e


#
def predictive_model( Features, Model ):
    """ input: a list of two features
                [month,day]
        output: the predicted popularity of birthdays:
                                 above or below median
    """
    our_features = np.asarray([Features])                 # extra brackets needed
    predicted_popularity = Model.predict(our_features)       # The model's prediction!
    predicted_popularity = int(round(predicted_popularity[0]))  # unpack the extra brackets
    return predicted_popularity

LoF = [
[4,9],
[9,9],
[2,31],
[12,70],
[14,8],
]

for Features in LoF:
    predicted_popularity = predictive_model( Features, rforest_model_tuned )  # pass in the model, too!
    name = POPULARITY[predicted_popularity]
    print(f"from the features {Features} I predict {name}")


VERTICAL = np.arange(0,12,1) # array of vertical input values
HORIZONT = np.arange(0,31,1) # array of horizontal input values
PLANE = np.zeros( (len(HORIZONT),len(VERTICAL)) ) # the output array
MODEL = rforest_model_tuned

col = 0
row = 0
for month in VERTICAL:
  for day in HORIZONT:
    Features = [month,day]
    output = predictive_model(Features,MODEL)
    PLANE[row,col] = output
    row += 1
  row = 0
  col += 1
  print(".", end="")

print("\n", PLANE[0:3,0:3])

import seaborn as sns

fig, ax = plt.subplots(figsize=(12,8))

im = ax.imshow(PLANE, cmap="viridis", extent=[HORIZONT.min(), HORIZONT.max(), VERTICAL.min(), VERTICAL.max()], origin="lower", aspect="auto")

ax.set_xlabel("month", fontsize=14)
ax.set_ylabel("day", fontsize=14)

ax.set_title(f"RF: Popularity of Birthdays \n", fontsize=18)

reduced_tick_indices = np.arange(0, len(HORIZONT), len(HORIZONT)//8)

# ax.set_xticks(HORIZONT[reduced_tick_indices]) # Display ticks every 0.4 unit
# ax.set_yticks(VERTICAL[reduced_tick_indices])
# ax.set_xticklabels([f"{x:.1f}" for x in HORIZONT[reduced_tick_indices]], fontsize=12)  # Format x-axis labels
# ax.set_yticklabels([f"{y:.1f}" for y in VERTICAL[reduced_tick_indices]], fontsize=12)  # Format y-axis labels

# Add a colorbar
cbar = plt.colorbar(im)
cbar.set_label('Predicted Species (0: Below, 1: Above)', rotation=270, labelpad=25)

plt.show()



# digits

import time
import sklearn
import numpy as np
import pandas as pd

cleaned_filename = "digits_cleaned.csv"
df_tidy = pd.read_csv(cleaned_filename)
df_model1 = df_tidy

# organizing... so fun... I'm very organized......
COLUMNS = df_model1.columns
DIGIT = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
DIDIT_INDEX = { c:i for i,c in enumerate(DIGIT) }
COLUMNS_INDEX = { c:i for i,c in enumerate(COLUMNS) }

# munpy! I mean-- numpy!!!
A = df_model1.to_numpy()
A = A.astype('float64')
NUM_ROWS, NUM_COLS = A.shape

# (features, what to predict)
X_all = A[:,0:64]
y_all = A[:,64]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20)
from sklearn import tree
best_depth = 3
dtree_model = tree.DecisionTreeClassifier(max_depth=best_depth)
dtree_model.fit(X_train, y_train)

# teeeeeeeeeeeeeeeest!!!!!!
predicted_labels = dtree_model.predict(X_test)
actual_labels = y_test
if False:
  print("Predicted labels:", predicted_labels)
  print("Actual  labels  :", actual_labels)
  num_correct = sum(predicted_labels == actual_labels)
  total = len(actual_labels)
  print(f"\nResults on test set:  {num_correct} correct out of {total} total, for {num_correct*100/total}%\n")

# cross multiply-- I mean validate! of course, haha...
from sklearn.model_selection import cross_val_score
all_accuracies = []
best_depth = 3
best_accuracy = 0.0
for depth in range(1,20):
  dtree_cv_model = tree.DecisionTreeClassifier(max_depth=depth)
  cv_scores = cross_val_score( dtree_cv_model, X_train, y_train, cv=5 )
  this_cv_accuracy = cv_scores.mean()
  if this_cv_accuracy > best_accuracy:
      best_accuracy = this_cv_accuracy
      best_depth = depth
  all_accuracies.append(this_cv_accuracy)
print(f"best_depth = {best_depth}  \n    yields the highest cv accuracy: {best_accuracy}\n")

# final!!! ...but the best depth keeps changing T v T
dtree_model_final = tree.DecisionTreeClassifier(max_depth=best_depth)
dtree_model_final.fit(X_all, y_all)
if True:  print("Created and trained a classifier with best_depth =", best_depth)
print("\nThe feature importances are", dtree_model_final.feature_importances_)



dtree_model_final


# T R E E (pt. 2)

import matplotlib.pyplot as plt
FEATURES = COLUMNS[0:64]

filename = 'tree_data2.gv'
model = dtree_model_final

tree.export_graphviz(model, out_file=filename,  # the filename constructed above...!
                            feature_names=COLUMNS[:-1], # actual feature names
                            filled=True,              # fun!
                            rotate=False,             # False for Up/Down; True for L/R
                            class_names=DIGIT,      # good to have
                            leaves_parallel=True )    # lots of options!

print(f"file {filename} written. Try pasting its contents to  http://viz-js.com/\n")

with open(filename, "r") as f:
    all_file_text = f.read()
    print(all_file_text)

#
# Tree display...
#
fig = plt.figure(figsize=(12,8))
tree_plot = tree.plot_tree(model,
                   feature_names=FEATURES,   # Glad to have these!
                   class_names=DIGIT,      # and these!!
                   filled=True)

plt.show()


def predictive_model( Features, Model ):
    """ input: a list of 64 features

        output: the predicted number, 0-9
    """
    our_features = np.asarray([Features])
    predicted_species = Model.predict(our_features)
    predicted_species = int(round(predicted_species[0]))
    return predicted_species

ListofFeatures = [ [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,1,4,6,7,2,1,0,0,1,5,1,3,6,7,0,0,5,16,14,12,10,2,3,4,0,0,3,0,16,0,14,0,1,0,0,0,12,1,2,4,6,7,8,9,0,0,0,6,7,0,0,0,12,14,0,0,0,8,9,0,0,0],
                   [0,2,1,3,2,4,3,5,4,6,5,7,6,8,7,9,8,10,9,11,10,12,13,15,0,2,1,3,2,4,3,5,4,6,5,7,6,8,7,9,8,10,9,11,10,12,13,15,0,2,1,3,2,4,3,5,4,6,5,7,6,8,7,9]
                 ]

for Features in ListofFeatures:
    predicted_number = predictive_model(Features, dtree_model_final)
    print(f"from the features {Features} \nI predict {predicted_number} ")


# the one that was predicted as 3

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
sns.set_theme()

apparently_3 = [0,0,1,4,6,7,2,1,0,0,1,5,1,3,6,7,0,0,5,16,14,12,10,2,3,4,0,0,3,0,16,0,14,0,1,0,0,0,12,1,2,4,6,7,8,9,0,0,0,6,7,0,0,0,12,14,0,0,0,8,9,0,0,0],


row_to_show = 42

pixels_as_row = apparently_3
print("pixels as 1d numpy array (row):\n", pixels_as_row)

pixels_as_image = np.reshape(apparently_3, (8,8))
print("\npixels as 2d numpy array (image):\n", pixels_as_image)

# create the figure, f, and the axes, ax:
f, ax = plt.subplots(figsize=(9, 7))

# colormap choice! Fun!
our_colormap = sns.color_palette("light:b", as_cmap=True)


# Draw a heatmap with the numeric values in each cell (make annot=False to remove the values)
sns.heatmap(pixels_as_image, annot=False, fmt="d", linewidths=.5, ax=ax, cmap=our_colormap)


# digits
# forests are cool. I think.

import time
import sklearn
import numpy as np
import pandas as pd

cleaned_filename = "digits_cleaned.csv"
df_tidy = pd.read_csv(cleaned_filename)
df_model1 = df_tidy

# organizing... so fun... I'm very organized......
COLUMNS = df_model1.columns
DIGIT = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
DIDIT_INDEX = { c:i for i,c in enumerate(DIGIT) }
COLUMNS_INDEX = { c:i for i,c in enumerate(COLUMNS) }

# munpy! I mean-- numpy!!!
A = df_model1.to_numpy()
A = A.astype('float64')
NUM_ROWS, NUM_COLS = A.shape

# (features, what to predict)
X_all = A[:,0:64]
y_all = A[:,64]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20)
from sklearn import tree
from sklearn import ensemble # Y E E E E L L L L L L L L L L L L below this changes in random forests ANKDDANKDNKDNNVKNDNK
best_d = 3
best_num_trees = 33
rforest_model = ensemble.RandomForestClassifier(max_depth=best_d, n_estimators=best_num_trees, max_samples=0.5)
rforest_model.fit(X_train, y_train)

# teeeeeeeeeeeeeeeest!!!!!!
predicted_labels = rforest_model.predict(X_test)
actual_labels = y_test
if False:
  print("Predicted labels:", predicted_labels)
  print("Actual  labels  :", actual_labels)
  num_correct = sum(predicted_labels == actual_labels)
  total = len(actual_labels)
  print(f"\nResults on test set:  {num_correct} correct out of {total} total, for {num_correct*100/total}%\n")

# cross multiply-- I mean validate! of course, haha...
from sklearn.model_selection import cross_val_score
all_accuracies = []
best_d = 3
best_num_trees = 50
best_accuracy = 0.0
for d in range(1,6):
  for ntrees in [50,100,150]:
    rforest_model = ensemble.RandomForestClassifier(max_depth=d, n_estimators=ntrees,max_samples=0.5)
    cv_scores = cross_val_score(rforest_model, X_train, y_train, cv=5)
    average_cv_accuracy = cv_scores.mean()
    if average_cv_accuracy > best_accuracy:
      best_accuracy = average_cv_accuracy;   best_d = d;      best_num_trees = ntrees
print(f"best_d: {best_d} and best_num_trees: {best_num_trees} are our choices. Acc: {best_accuracy}")

rforest_model_tuned = ensemble.RandomForestClassifier(max_depth=best_d, n_estimators=best_num_trees, max_samples=0.5)
rforest_model_tuned.fit(X_all, y_all)
print("\nThe feature importances are", rforest_model_tuned.feature_importances_)


#
# we can get the individual trees, if we want...  Let's try it on tree #72
#
tree_index = 72   # which tree
one_rf_tree = rforest_model_tuned.estimators_[tree_index]
print(f"One of the forest's trees is {one_rf_tree}")

# From there, it's possible to create a graphical version...
filename = f'rf_tree_{tree_index:03d}.gv'             # f strings! Could save all trees, but we won't do so here.
tree.export_graphviz(one_rf_tree, out_file=filename,  # the filename constructed above...!
                            feature_names=FEATURES, # actual feature names, not species
                            filled=True,              # fun!
                            rotate=False,             # False for Up/Down; True for L/R
                            class_names=DIGIT,      # good to have
                            leaves_parallel=True )    # lots of options!

print(f"file {filename} written. Try copying the result to http://viz-js.com/ \n")
with open(filename, "r") as f:
    file_text = f.read()
    print(file_text)

# One tree:
fig = plt.figure(figsize=(10,8))
tree_plot = tree.plot_tree(one_rf_tree,
                   feature_names=FEATURES,   # Glad to have these!
                   class_names=DIGIT,      # and these!!
                   filled=True)


#
def predictive_model( Features, Model ):
    """ input: a list of 64 features

        output: the predicted number, 0-9
    """
    our_features = np.asarray([Features])                 # extra brackets needed
    predicted_digit = Model.predict(our_features)       # The model's prediction!
    predicted_digit = int(round(predicted_digit[0]))  # unpack the extra brackets
    return predicted_digit

LoF = [
[0,0,1,4,6,7,2,1,0,0,1,5,1,3,6,7,0,0,5,16,14,12,10,2,3,4,0,0,3,0,16,0,14,0,1,0,0,0,12,1,2,4,6,7,8,9,0,0,0,6,7,0,0,0,12,14,0,0,0,8,9,0,0,0]
]

for Features in LoF:
    predicted_digit = predictive_model( Features, rforest_model_tuned )  # pass in the model, too!
    name = DIGIT[predicted_digit]
    print(f"from the features {Features} \nI predict {name}")


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
sns.set_theme()

digits_cleaned = pd.read_csv('digits_cleaned.csv')
list_of_column_names = digits_cleaned.columns

ROW = 0
COLUMN = 1
digitsA = digits_cleaned.values

row_to_show = 110

pixels_as_row = digitsA[row_to_show,0:64]
print("pixels as 1d numpy array (row):\n", pixels_as_row)

pixels_as_image = np.reshape(pixels_as_row, (8,8))
print("\npixels as 2d numpy array (image):\n", pixels_as_image)

# create the figure, f, and the axes, ax:
f, ax = plt.subplots(figsize=(9, 7))

# colormap choice! Fun!
our_colormap = sns.color_palette("light:b", as_cmap=True)


# Draw a heatmap with the numeric values in each cell (make annot=False to remove the values)
sns.heatmap(pixels_as_image, annot=True, fmt="d", linewidths=.5, ax=ax, cmap=our_colormap)


