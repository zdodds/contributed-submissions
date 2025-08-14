#
# coding cell to make sure Python is happy...

14*3


# libraries...
import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)


filename = 'digits_cleaned.csv' # neighborhoods
df_tidy = pd.read_csv(filename)      # encoding = "utf-8", "latin1"
print(f"{filename} : file read into a pandas dataframe.")


print(f"df_tidy.shape is {df_tidy.shape}\n")
df_tidy.info()  # prints column information
df_tidy


df_tidy_cat = pd.get_dummies(data=df_tidy,prefix="is",columns=['actual_digit'])
df_tidy_cat


ROW = 0
COLUMN = 1
df_model1 = df_tidy_cat.drop('numbernum', axis=COLUMN )
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

DIGITS = ['0','1','2','3','4','5','6','7','8','9']   # int to str
DIGITS_INDEX = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}  # str to int

# Let's try it out...
for name in DIGITS:
    print(f"{name} maps to {DIGITS_INDEX[name]}")


A = df_model1.to_numpy()   
A = A.astype('float64')    # many types:  www.tutorialspoint.com/numpy/numpy_data_types.htm
print(A[:,:])


NUM_ROWS, NUM_COLS = A.shape
print(f"\nThe dataset has {NUM_ROWS} rows and {NUM_COLS} cols")


# let's use all of our variables, to reinforce that we have
# (1) names...
# (2) access and control...

# choose a row index, n:
n = 42
print(f"Digit #{n} is {A[n]}")
digits = ""

for i in range(len(COLUMNS)):
    colname = COLUMNS[i]
    value = A[n][i]
    print(f"  Its {colname} is {value}")
    if "0" in colname: digits = "0"
    elif "1" in colname: digits = "1"
    elif "2" in colname: digits = "2"
    elif "3" in colname: digits = "3"
    elif "4" in colname: digits = "4"
    elif "5" in colname: digits = "5"
    elif "6" in colname: digits = "6"
    elif "7" in colname: digits = "7"
    elif "8" in colname: digits = "8"
    elif "9" in colname: digits = "9"

print()
print(f"  and its digit name: {digits}")


print("+++ Start of data definitions +++\n")

#
# we could do this at the data-frame level, too!
#

X_all = A[:,0:64]   # X (features) ... is all rows, columns 0, 1, 2, 3
y_all = A[:,64:]    # y (labels) ... is all rows, columns 4, 5, and 6

print(f"y_all (just the labels/species, first few rows) are \n {y_all[0:5]}")
print()
print(f"X_all (just the features, first few rows) are \n {X_all[0:5]}")


#
# we can scramble the data, to remove (potential) dependence on its ordering: 
# 
indices = np.random.permutation(len(y_all))  # indices is a permutation-list

# we scramble both X and y, necessarily with the same permutation
X_all = X_all[indices]              # we apply the _same_ permutation to each!
y_all = y_all[indices]              # again...
print(f"The scrambled labels/species are \n {y_all[0:5]}")
print()
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

print(f"+++ Testing +++   Held-out data... (testing data: {len(y_test)})\n")
print(f"y_test: {y_test[0:5,:]}\n")
print(f"X_test (few rows): {X_test[0:5,:]}")  # 5 rows
print("\n")
print(f"+++ Training +++   Data used for modeling... (training data: {len(y_train)})\n")
print(f"y_train: {y_train[0:5,:]}\n")
print(f"X_train (few rows): {X_train[0:5,:]}")  # 5 rows


#
# for NNets, it's important to keep the feature values near 0, say -1. to 1. or so
#    This is done through the "StandardScaler" in scikit-learn
# 
from sklearn.preprocessing import StandardScaler

#
# do we want to use a Scaler?
#
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

y_train_scaled = y_train.copy()  # the predicted/desired labels are not scaled
y_test_scaled = y_test.copy()  # not using the scaler




#
# let's create a table for showing our data and its predictions...
#
def ascii_table(X,y,scaler_to_invert=None):
    """ print a table of inputs and outputs """
    np.set_printoptions(precision=2)  # Let's use less precision
    if scaler_to_invert == None:  # don't use the scaler
        X = X
    else:
        X = scaler_to_invert.inverse_transform(X)
    print(f"{'input ':>58s} -> {'pred':^7s} {'des':<5s}") 
    for i in range(len(y)):
        # whoa! serious f-string formatting:
        print(f"{str(X[i,0:64]):>58s} -> {'?':^7s} {str(y[i]):<21s}")   # !s is str ...
    print()
    
# to show the table with the scaled data:
ascii_table(X_train_scaled[0:5,:],y_train_scaled[0:5],None)

# to show the table with the original data:
ascii_table(X_train_scaled[0:5,:],y_train_scaled[0:5],scaler_to_invert=scaler)


from sklearn.neural_network import MLPClassifier

#
# Here's where you can change the number of hidden layers
# and number of neurons!  It's in the tuple  hidden_layer_sizes:
#
nn_classifier = MLPClassifier(hidden_layer_sizes=(6,7),  
                    # hidden_layer_sizes=(6,7)   means   4 inputs -> 6 hidden -> 7 hidden -> 3 outputs
                    max_iter=500,      # how many times to train
                    # activation="tanh", # the "activation function" input -> output
                    # solver='sgd',      # the algorithm for optimizing weights
                    verbose=True,      # False to "mute" the training
                    shuffle=True,      # reshuffle the training epochs?
                    random_state=None, # set for reproduceability
                    learning_rate_init=.1,       # learning rate: the amt of error to backpropagate!
                    learning_rate = 'adaptive')  # soften feedback as it converges

# documentation:
# scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html 
#     Try other network sizes / other parameters ...

print("\n\n++++++++++  TRAINING:  begin  +++++++++++++++\n\n")
nn_classifier.fit(X_train_scaled, y_train_scaled)
print("\n++++++++++  TRAINING:   end  +++++++++++++++")
print(f"The analog prediction error (the loss) is {nn_classifier.loss_}")


L=  [9.74e-22, 2.96e-06, 4.22e-17, 1.63e-03, 1.71e-05, 8.42e-04, 3.41e-20, 2.58e-01,
 9.60e-04, 1.81e-04]
get_digits(L)


#
# how did it do on the testing data?
#

def get_digits(A):
    m = max(A)
    for i in range(len(A)):
        if A[i] == m: 
            return DIGITS[i]  # note that this "takes the first one"
    return "no digits" 

SEE_PROBS = False

def ascii_table_for_classifier(Xsc,y,nn,scaler):
    """ a table including predictions using nn.predict """
    predictions = nn.predict(Xsc)            # all predictions
    prediction_probs = nn.predict_proba(Xsc) # all prediction probabilities
    Xpr = scaler.inverse_transform(Xsc)      # Xpr is the "X to print": the unscaled data
    # count correct
    num_correct = 0
    # printing
    print(f"{'input ':>28s} -> {'pred':^12s} {'des.':^12s}") 
    for i in range(len(y)):
        pred = predictions[i]
        pred_probs = prediction_probs[i,:]
        desired = y[i].astype(int)
        # print(pred, desired, pred_probs)
        pred_digits = get_digits(pred_probs)
        des_digits  = get_digits(desired)
        if pred_digits != des_digits: result = "  incorrect: " + str(pred_probs)
        else: result = "  correct" + (": "+pred_probs if SEE_PROBS else "") ; num_correct += 1
        # Xpr = Xsc  # if you want to see the scaled versions
        print(f"{Xpr[i,0:65]!s:>28s} -> {pred_digits:^12s} {des_digits:12s} {result:^10s}") 
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
    np.set_printoptions(precision=2)  # Let's use less precision
    nn = nn_classifier  # less to type?
    print("\n\n+++++ parameters, weights, etc. +++++\n")
    print(f"\nweights/coefficients:\n")
    for i, wts in enumerate(nn.coefs_):
        print(f"[[ Layer {i} ]]\n   has shape = {wts.shape} and weights =\n{wts}")
        print(f"   with intercepts:\n {nn.intercepts_[i]}\n")
    print()
    print(f"\nall parameters: {nn.get_params()}")


for sublist in X_test:
    print(' '.join(str(item) for item in sublist))



int(np.argmax([0,0,0,0,.0001,0,0,0,0]))


#
# final predictive model (random forests), with tuned parameters + ALL data incorporated
#

def get_species(A):
    """ returns the species for A ~ [1 0 0] or [0 1 0] or ... """
    for i in range(len(DIGITS)):
        if A[i] == 1: 
            return DIGITS[i]  # note that this "takes the first one"
    return "no species in get_species" 


def predictive_model( Features, MODEL, SCALER ):
    """ input: a list of four features 
                [ sepallen, sepalwid, petallen, petalwid ]
        output: the predicted species of iris, from
                  setosa (0), versicolor (1), virginica (2)
    """
    our_features = np.asarray([Features])                 # extra brackets needed
    scaled_features = SCALER.transform(our_features)      # we have to scale the features into "scaled space"
    predicted_cat = MODEL.predict(scaled_features)        # then, the nnet can predict our "cat" variables
    prediction_probs = MODEL.predict_proba(scaled_features) # all prediction probabilities
    # our_features = SCALER.inverse_transform(scaled_features)  # we can convert back (optional!)
    #print(predicted_cat[0])
    x = int(np.argmax(prediction_probs))
    #predicted_species = get_species(predicted_cat[0])     # (it's extra-nested) get the species name
    predicted_species = str(x)
    return predicted_species, prediction_probs
   
#
# Try it!
# 
# Features = eval(input("Enter new Features: "))
#
Features = [6.7,3.3,5.7,0.1]  # [5.8,2.7,4.1,1.0] [4.6,3.6,3.0,2.2] [6.7,3.3,5.7,2.1]

LoF = [
[0.0, 0.0, 7.0, 13.0, 11.0, 1.0, 0.0, 0.0, 0.0, 6.0, 14.0, 12.0, 14.0, 9.0, 0.0, 0.0, 0.0, 5.0, 14.0, 3.0, 10.0, 9.0, 0.0, 0.0, 0.0, 0.0, 8.0, 15.0, 14.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 14.0, 16.0, 6.0, 0.0, 0.0, 0.0, 0.0, 9.0, 9.0, 3.0, 15.0, 4.0, 0.0, 0.0, 0.0, 12.0, 5.0, 1.0, 11.0, 8.0, 0.0, 0.0, 0.0, 7.0, 16.0, 16.0, 9.0, 1.0, 0.0],   
[0.0, 0.0, 3.0, 13.0, 16.0, 7.0, 0.0, 0.0, 0.0, 1.0, 12.0, 16.0, 8.0, 0.0, 0.0, 0.0, 0.0, 2.0, 16.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 16.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 16.0, 10.0, 11.0, 4.0, 0.0, 0.0, 0.0, 6.0, 16.0, 14.0, 13.0, 16.0, 3.0, 0.0, 0.0, 1.0, 11.0, 11.0, 2.0, 14.0, 10.0, 0.0, 0.0, 0.0, 2.0, 15.0, 16.0, 15.0, 6.0, 0.0],   
[0.0, 0.0, 4.0, 15.0, 16.0, 16.0, 16.0, 1.0, 0.0, 0.0, 10.0, 13.0, 8.0, 15.0, 8.0, 0.0, 0.0, 0.0, 14.0, 5.0, 3.0, 16.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 12.0, 11.0, 0.0, 0.0, 0.0, 0.0, 2.0, 5.0, 16.0, 9.0, 1.0, 0.0, 0.0, 0.0, 15.0, 16.0, 16.0, 14.0, 3.0, 0.0, 0.0, 0.0, 1.0, 15.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 14.0, 2.0, 0.0, 0.0, 0.0],   # actually virginica
[0.0, 0.0, 0.0, 1.0, 7.0, 14.0, 14.0, 0.0, 0.0, 0.0, 3.0, 15.0, 7.0, 1.0, 14.0, 0.0, 0.0, 2.0, 16.0, 10.0, 5.0, 14.0, 8.0, 0.0, 0.0, 4.0, 15.0, 16.0, 12.0, 16.0, 5.0, 0.0, 0.0, 0.0, 5.0, 3.0, 1.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 12.0, 0.0, 0.0]
]

SEE_PROBS = True

# run on each one:
for Features in LoF:
    MODEL = nn_classifier
    SCALER = scaler
    name, probs = predictive_model( Features, MODEL, SCALER )  # pass in the model, too!
    prob_str = "   with probs: " + str(probs) if SEE_PROBS == True else ""
    print(f"I predict {name:>12s} from the features {Features}  {prob_str}")    # Answers in the assignment...


#
# What shall we predict today?
print(COL_INDEX)
print()
print(COLUMNS)




#
# Here we set up for a regression model that will predict 'sepallen'  (column index 0)
#
#   sepal length  'sepallen' (column index 0)  <-- our target for _regression_

# We will use
#
#   sepal width   'sepalwid' (column index 1)
#   petal length  'petallen' (column index 2)
#   petal width   'petalwid' (column index 3)
#   _and_
#   species       'irisnum'  (column index 4)    # No problem - we can use this as a feature!

print("+++ Start of data-assembly for feature-regression! +++\n")
# construct the correct X_all from the columns we want
# we use np.concatenate to combine parts of the dataset to get all-except-column 0:
#                     exclude 0  , include 1 to the end
for i in range(64):
    X_all = np.concatenate( (A[:,0:i], A[:,i+1:]), axis=1)    # includes columns 1, 2, 3, and 4

# if we wanted all-except-column 1:   X_all = np.concatenate( (A[:,0:1], A[:,2:]),axis=1)  # columns 0, 2, 3, and 4
# if we wanted all-except-column 2:   X_all = np.concatenate( (A[:,0:2], A[:,3:]),axis=1)  # columns 0, 1, 3, and 4
# if we wanted all-except-column 3:   X_all = np.concatenate( (A[:,0:3], A[:,4:]),axis=1)  # columns 0, 1, 2, and 4
# if we wanted all-except-column 4:   X_all = np.concatenate( (A[:,0:4], A[:,5:]),axis=1)  # columns 0, 1, 2, and 3
# (slicing is forgiving) 


    y_all = A[:,i]                    # y (labels) ... is all of column 0, sepallen (sepal length) 
#                                 # change the line above to make other columns the target (y_all)
print(f"y_all is \n {y_all}")
print() 
print(f"X_all (the features, first few rows) is \n {X_all[:5,:]}")



#
# Here we set up for a regression model that will predict 'sepallen'  (column index 0)
#
#   sepal length  'sepallen' (column index 0)  <-- our target for _regression_

# We will use
#
#   sepal width   'sepalwid' (column index 1)
#   petal length  'petallen' (column index 2)
#   petal width   'petalwid' (column index 3)
#   _and_
#   species       'irisnum'  (column index 4)    # No problem - we can use this as a feature!

print("+++ Start of data-assembly for feature-regression! +++\n")
# construct the correct X_all from the columns we want
# we use np.concatenate to combine parts of the dataset to get all-except-column 0:
#                     exclude 0  , include 1 to the end
X_all = np.concatenate( (A[:,0:42], A[:,43:]), axis=1)    # includes columns 1, 2, 3, and 4

# if we wanted all-except-column 1:   X_all = np.concatenate( (A[:,0:1], A[:,2:]),axis=1)  # columns 0, 2, 3, and 4
# if we wanted all-except-column 2:   X_all = np.concatenate( (A[:,0:2], A[:,3:]),axis=1)  # columns 0, 1, 3, and 4
# if we wanted all-except-column 3:   X_all = np.concatenate( (A[:,0:3], A[:,4:]),axis=1)  # columns 0, 1, 2, and 4
# if we wanted all-except-column 4:   X_all = np.concatenate( (A[:,0:4], A[:,5:]),axis=1)  # columns 0, 1, 2, and 3
# (slicing is forgiving) 

y_all = A[:,42]                    # y (labels) ... is all of column 0, sepallen (sepal length) 
#                                 # change the line above to make other columns the target (y_all)
print(f"y_all is \n {y_all}")
print() 
print(f"X_all (the features, first few rows) is \n {X_all[:5,:]}")



#
# we scramble the data, to give a different TRAIN/TEST split each time...
# 
indices = np.random.permutation(len(y_all))  # indices is a permutation-list

# we scramble both X and y, necessarily with the same permutation
X_all = X_all[indices]              # we apply the _same_ permutation to each!
y_all = y_all[indices]              # again...
print("target values to predict: \n",y_all)
print("\nfeatures (a few)\n", X_all[0:5,:])


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
                    max_iter=500,          # how many training epochs
                    verbose=True,          # do we want to watch as it trains?
                    shuffle=True,          # shuffle each epoch?
                    random_state=None,     # use for reproducibility
                    learning_rate_init=.1, # how much of each error to back-propagate
                    learning_rate = 'adaptive')  # how to handle the learning_rate

print("\n\n++++++++++  TRAINING:  begin  +++++++++++++++\n\n")
nn_regressor.fit(X_train_scaled, y_train_scaled)
print("++++++++++  TRAINING:   end  +++++++++++++++")
print(f"The (squared) prediction error (the loss) is {nn_regressor.loss_:<6.3f}")
print(f"And, its square root:         {nn_regressor.loss_ ** 0.5:<6.3f}")
print()


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
    print(f"{'input ':>35s} ->  {'pred':^6s}  {'des.':^6s}   {'absdiff':^10s}") 
    for i in range(len(y)):
        pred = predictions[i]
        desired = y[i]
        result = abs(desired - pred)
        error += result
        # Xpr = Xsc   # if you'd like to see the scaled values
        print(f"{Xpr[i,0:2]!s:>35s} ->  {pred:<+6.2f}  {desired:<+6.2f}   {result:^10.2f}") 
    for i in range(len(y)):
        pred = predictions[i]
        desired = y[i]
        result = abs(desired - pred)
        error += result
        # Xpr = Xsc   # if you'd like to see the scaled values
        print("The predicted pixel is "f"{pred:<+6.2f}") 

    print("\n" + "+++++   +++++   +++++           ")
    print(f"average abs diff error:   {error/len(y):<6.3f}")
    print("+++++   +++++   +++++           ")
    
#
# let's see how it did on the test data (also the training data!)
#
ascii_table_for_regressor(X_test_scaled,
                          y_test_scaled,
                          nn_regressor,
                          scaler)   # this is our own f'n, above
print(f"{'des.':^6s}")



