#
# hw5pr2digits_modeler:  handwritten-digit modeling with k-nearest neighbors...
#


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
sns.set_theme()

# Load the example flights dataset and convert to CS35_Participant_2-form
digits_orig = pd.read_csv('./digits.csv')
list_of_column_names = digits_orig.columns

ROW = 0
COLUMN = 1
digits2 = digits_orig.drop( list_of_column_names[-1], axis=COLUMN)  # drop the rightmost column - it's just a url!
digitsA = digits2.values  # get a numpy array (digitsA) from the dataframe (digits2)

row_to_show = 42   # choose the digit (row) you want to show... (watch out for indexing!)

pixels_as_row = digitsA[row_to_show,0:64]
print("pixels as 1d numpy array (row):\n", pixels_as_row)

pixels_as_image = np.reshape(pixels_as_row, (8,8))   # reshape into a 2d 8x8 array (image)
print("\npixels as 2d numpy array (image):\n", pixels_as_image)

# create the figure, f, and the axes, ax:
f, ax = plt.subplots(figsize=(9, 6))

# colormap choice! Fun!   www.practicalpythonfordatascience.com/ap_seaborn_palette or seaborn.pydata.org/tutorial/color_palettes.html
our_colormap = sns.color_palette("light:b", as_cmap=True) 

# Draw a heatmap with the numeric values in each cell (make annot=False to remove the values)
sns.heatmap(pixels_as_image, annot=True, fmt="d", linewidths=.5, ax=ax, cmap=our_colormap)


#
# hw5pr1iris_modeler:  iris clasification via nearest neighbors
#

# We will be using the sklearn library - let's check if we have it:
import sklearn


# libraries!
import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)


# let's read in our flower data...
# 
cleaned_filename = "digits_cleaned.csv"
df_tidy = pd.read_csv(cleaned_filename)   # encoding="utf-8" et al.
print(f"{cleaned_filename} : file read into a pandas dataframe.")
#df_tidy1 = df_tidy.drop('actual_digit',axis=COLUMN)
#df_tidy1


#
# it's no fun fiddling with the default table formatting.
#
# Remember: we have the data itself!  If we want to see it, we can print:

(NROWS, NCOLS) = df_tidy.shape
print(f"There are {NROWS = } and {NCOLS = }")
print()

for row in range(0,NROWS,5):
    print(df_tidy[row:row+5])    # Let's print 5 at a time...


#
# once we have all the columns we want, let's create an index of their names...

#
# Let's make sure we have all of our helpful variables in one place 
#       To be adapted if we drop/add more columns...
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

DIGITS = ['0','1','2','3','4','5','6','7','8','9']   # int to str
DIGITS_INDEX = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}  # str to int


A = df_tidy.to_numpy()    # yields the underlying numpy array
print(A)




#
# let's make sure it's all floating-point, so we can multiply and divide
#
#       this is not needed here, but it can be important if some features are integer and floating point is needed


A = A.astype('float64')  # so many numpy types!   Here is a list:  www.tutorialspoint.com/numpy/numpy_data_types.htm
print(A)



#
# Also, nice to have NUM_ROWS and NUM_COLS around
#
NUM_ROWS, NUM_COLS = A.shape
print(f"\nThe dataset has {NUM_ROWS} rows and {NUM_COLS} cols")


# let's use all of our variables, to reinforce that we have everything:
# (1) names...
# (2) access and control...
# choose a row index, n:

n = 42     # the row number

# This shows us that we have complete access to any individual data "point" (row)
print(f"pixels #{n} is {A[n]}")

for i in range(len(COLUMNS)):
    colname = COLUMNS[i]
    value = A[n][i]
    print(f"  Its {colname} is {round(value,2)}")



df_tidy.shape


df_tidy.columns


print("+++ Start of data definitions +++\n")

#
# we could do this at the data-frame level, too!
#

#
# Watch out!  Between datasets, this cell is one that often needs to be carefully changed...
#

X_all = A[:,0:64]  # X (features) ... is all rows, columns 0, 1, 2, 3
y_all = A[:,64]    # y (labels) ... is all rows, column 4 only

print(f"y_all (actual_digits)   are all here: \n {y_all}")
print()
print(f"X_all (just the features, 5 rows worth) are \n {X_all[0:5]}")



#
# we scramble the data, to remove (potential) dependence on the data ordering:
# 
indices = np.random.permutation(len(y_all))  # indices is a permutation-list

# we scramble both X and y, necessarily with the same permutation
X_labeled = X_all[indices]              # we apply the _same_ permutation to each!
y_labeled = y_all[indices]              # again...
print(f"The scrambled labels/species are \n {y_labeled}")
print()
print(f"The corresponding data rows are \n {X_labeled}")



#
# We next separate into test data and training data ... 
#    + We will train on the training data...
#    + We will _not_ look at the testing data at all when building the model
#
# Then, afterward, we will test on the testing data -- and see how well we do!
#

#
# a common convention:  train on 80%, test on 20%    To do so, let's define TEST_PERCENT as 0.2
#

TEST_PERCENT = 0.2

from sklearn.model_selection import train_test_split      # this function splits into training + testing sets

# Here we create four numpy arrays:
#    X_train are a 2d array of features and observations for training
#    y_train are a single-column of the correct species for X_train (that's how it trains!)
#
#    X_test are a 2d array of features and observations for testing (unseen during training)
#    y_test are a single-column of the correct species for X_test (so we can measure how well the testing goes...) 

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=TEST_PERCENT)  # random_state=42

# Done!  Let's confirm these match our intution:

print(f"training with {len(y_train)} rows;  testing with {len(y_test)} rows\n" )

print(f"Held-out data... (testing data: {len(y_test)} rows)")
print(f"y_test: {y_test}")
print(f"X_test (first few rows): {X_test[0:5,:]}")  # 5 rows
print()
print(f"Data used for modeling... (training data: {len(y_train)} rows)")
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

# we train the model ... it's one line!
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

        print(f"row {i:>3d} : {DIGITS[p]:>12s} {DIGITS[a]:<12s}   {result}")   

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
    predicted_digits_list = knn_model.predict(our_features)   # PREDICT!

    predicted_digits= int(round(predicted_digits_list[0]))  # unpack the one element it contains
    name = DIGITS[predicted_digits]                          # look up the species
    return name        
    
#
# Try it!
# 
# Features = eval(input("Enter new Features: "))
#


LoD = [[0,0,0,8,14,0,0,0,0,0,5,16,11,0,0,0,0,1,15,14,1,6,0,0,0,7,16,5,3,16,8,0,0,8,16,8,14,16,2,0,0,0,6,14,16,11,0,0,0,0,0,6,16,4,0,0,0,0,0,10,15,0,0,0],
[0,0,0,5,14,12,2,0,0,0,7,15,8,14,4,0,0,0,6,2,3,13,1,0,0,0,0,1,13,4,0,0,0,0,1,11,9,0,0,0,0,8,16,13,0,0,0,0,0,5,14,16,11,2,0,0,0,0,0,6,12,13,3,0],
[0,0,0,3,16,3,0,0,0,0,0,12,16,2,0,0,0,0,8,16,16,4,0,0,0,7,16,15,16,12,11,0,0,8,16,16,16,13,3,0,0,0,0,7,14,1,0,0,0,0,0,6,16,0,0,0,0,0,0,4,14,0,0,0],
[0,0,0,3,15,10,1,0,0,0,0,11,10,16,4,0,0,0,0,12,1,15,6,0,0,0,0,3,4,15,4,0,0,0,0,6,15,6,0,0,0,4,15,16,9,0,0,0,0,0,13,16,15,9,3,0,0,0,0,4,9,14,7,0],
[0,0,0,3,16,3,0,0,0,0,0,10,16,11,0,0,0,0,4,16,16,8,0,0,0,2,14,12,16,5,0,0,0,10,16,14,16,16,11,0,0,5,12,13,16,8,3,0,0,0,0,2,15,3,0,0,0,0,0,4,12,0,0,0],
[0,0,7,15,15,4,0,0,0,8,16,16,16,4,0,0,0,8,15,8,16,4,0,0,0,0,0,10,15,0,0,0,0,0,1,15,9,0,0,0,0,0,6,16,2,0,0,0,0,0,8,16,8,11,9,0,0,0,9,16,16,12,3,0]]


for Features in LoD:
    result = predictive_model( Features )
    print(f"From the Features {Features}, I predict {result}")


#
# Here, we use "cross validation" to find the "best" k...
#

import time
from sklearn.model_selection import cross_val_score

#
# cross-validation splits the training set into two pieces:
#   + model-building and model-validation. We'll use "build" and "validate"
#
all_accuracies = []
best_k = 84  # Not correct!
best_accuracy = 0.0  # also not correct...

# Note that we are cross-validating using only our TEST data!
for k in range(1,85):
    knn_cv_model = KNeighborsClassifier(n_neighbors=k)   # build a knn_model for every k
    cv_scores = cross_val_score( knn_cv_model, X_train, y_train, cv=5 )  # cv=5 means 80/20
    this_cv_accuracy = cv_scores.mean()               # mean() is numpy's built-in average function 
    print(f"k: {k:2d}  cv accuracy: {this_cv_accuracy:7.4f}")

    if this_cv_accuracy > best_accuracy:  # is this one better?
        best_accuracy = this_cv_accuracy  # track the best accuracy
        best_k = k                        # with the best k

    all_accuracies.append(this_cv_accuracy)
    time.sleep(0.002)   # dramatic pauses!

    
# use best_k!
print(f"best_k = {best_k}   yields the highest average cv accuracy: {best_accuracy}")  # print the best one



### Let's see all the accuracies!

import pandas as pd

# Let's create a new pandas dataframe using the data from the above cell
crossvalidation_df = pd.DataFrame( {"k_value":np.asarray(range(1,84+1)),
                                    "accuracy":np.asarray(all_accuracies)}
                                    )

import seaborn as sns
sns.set_theme(style="whitegrid", rc = {'figure.figsize':(12,8)})  # other options: darkgrid, whitegrid, dark, white, ticks
sns.lineplot(x="k_value", y="accuracy", data=crossvalidation_df)


#
# With the best k, we build and train a new model:
#
# Now using best_k instead of the original, randomly-guessed value:   
#
best_k = best_k   # not needed, but nice to remind ourselves of the variable name
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
    predicted_digits = Model.predict(our_features)       # The model's prediction!
    predicted_digits = int(round(predicted_digits[0]))  # unpack the extra brackets
    return predicted_digits
   
#
# Try it!
# 

LoD = [[0,0,0,8,14,0,0,0,0,0,5,16,11,0,0,0,0,1,15,14,1,6,0,0,0,7,16,5,3,16,8,0,0,8,16,8,14,16,2,0,0,0,6,14,16,11,0,0,0,0,0,6,16,4,0,0,0,0,0,10,15,0,0,0],
[0,0,0,5,14,12,2,0,0,0,7,15,8,14,4,0,0,0,6,2,3,13,1,0,0,0,0,1,13,4,0,0,0,0,1,11,9,0,0,0,0,8,16,13,0,0,0,0,0,5,14,16,11,2,0,0,0,0,0,6,12,13,3,0],
[0,0,0,3,16,3,0,0,0,0,0,12,16,2,0,0,0,0,8,16,16,4,0,0,0,7,16,15,16,12,11,0,0,8,16,16,16,13,3,0,0,0,0,7,14,1,0,0,0,0,0,6,16,0,0,0,0,0,0,4,14,0,0,0],
[0,0,0,3,15,10,1,0,0,0,0,11,10,16,4,0,0,0,0,12,1,15,6,0,0,0,0,3,4,15,4,0,0,0,0,6,15,6,0,0,0,4,15,16,9,0,0,0,0,0,13,16,15,9,3,0,0,0,0,4,9,14,7,0],
[0,0,0,3,16,3,0,0,0,0,0,10,16,11,0,0,0,0,4,16,16,8,0,0,0,2,14,12,16,5,0,0,0,10,16,14,16,16,11,0,0,5,12,13,16,8,3,0,0,0,0,2,15,3,0,0,0,0,0,4,12,0,0,0],
[0,0,7,15,15,4,0,0,0,8,16,16,16,4,0,0,0,8,15,8,16,4,0,0,0,0,0,10,15,0,0,0,0,0,1,15,9,0,0,0,0,0,6,16,2,0,0,0,0,0,8,16,8,11,9,0,0,0,9,16,16,12,3,0]]


# run on each one:
for Features in LoD:
    predicted_digits = predictive_model( Features, knn_model_final )  # pass in the model, too!
    name = DIGITS[predicted_digits]
    print(f"I predict {name} from the features {Features}")    # Answers in the assignment...


#
# SUGGESTION:  
# 
# +++ copy-paste-and-alter from the iris- or births modeling notebooks into here +++
#
# when the data is ready to view, you might want to grab
# the digits-visualization code
#


target_column = 64
accuracy_results = []
best_k_results = []

for num_features in range(1, 65):

    X_all_part = A[:, 0:num_features]  
    y_all_part = A[:, target_column]

    # Cross-validation to find the best k
    best_k = -1
    best_accuracy = 0.0

    for k in range(1, 85):  # Loop over k values from 1 to 84
        knn_cv_model = KNeighborsClassifier(n_neighbors=k)  # Build a KNN model for each k
        cv_scores = cross_val_score(knn_cv_model, X_all_part, y_all_part, cv=5)  # Perform cross-validation
        this_cv_accuracy = cv_scores.mean()  # Compute the mean accuracy

        if this_cv_accuracy > best_accuracy:  # Update best accuracy and k
            best_accuracy = this_cv_accuracy
            best_k = k

    # Print the best k and corresponding accuracy for this num_features
    #print(f"For num_features = {num_features}, best_k = {best_k}, best CV accuracy: {best_accuracy:.4f}")
    print(f"n: {num_features:2d}  cv accuracy: {best_accuracy:7.4f}")


target_column = 64
accuracy_results = []
best_k_results = []
n_features = []

for num_features in range(1, 65):

    X_all_part = A[:, 0:num_features]  
    y_all_part = A[:, target_column]

    # Cross-validation to find the best k
    best_k = -1
    best_accuracy = 0.0

    for k in range(1, 85):  # Loop over k values from 1 to 84
        knn_cv_model = KNeighborsClassifier(n_neighbors=k)  # Build a KNN model for each k
        cv_scores = cross_val_score(knn_cv_model, X_all_part, y_all_part, cv=5)  # Perform cross-validation
        this_cv_accuracy = cv_scores.mean()  # Compute the mean accuracy

        if this_cv_accuracy > best_accuracy:  # Update best accuracy and k
            best_accuracy = this_cv_accuracy
            best_k = k

    accuracy_results.append(best_accuracy)
    best_k_results.append(best_k)
    n_features.append(num_features)

    # Print the best k and corresponding accuracy for this num_features
    #print(f"For num_features = {num_features}, best_k = {best_k}, best CV accuracy: {best_accuracy:.4f}")
    #print(f"n: {num_features:2d}  cv accuracy: {best_accuracy:7.4f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(n_features, accuracy_results, marker='o', linestyle='-')
plt.xlabel("Number of Features (n)")
plt.ylabel("Accuracy")

plt.show()


