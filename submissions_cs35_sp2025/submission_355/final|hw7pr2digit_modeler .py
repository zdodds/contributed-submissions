#
# coding cells, for copy-paste-and-adapt...

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
df_model1 = df_tidy_cat #.drop('actual_digit', axis=COLUMN )
df_model1


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


FEATURES = COLUMNS[0:64]

#
# and our "species" names
#

# all of scikit-learn's ML routines need numbers, not strings
#   ... even for categories/classifications (like species!)
#   so, we will convert the flower-species to numbers:

SPECIES = ['is_0','is_1', 'is_2', 'is_3', 'is_4', 'is_5', 'is_6', 'is_7', 'is_8', 'is_9']   # int to str
SPECIES_INDEX = {'is_0':0,'is_1':1, 'is_2':2, 'is_3':3, 'is_4':4, 'is_5':5, 'is_6':6, 'is_7':7, 'is_8':8, 'is_9':9}  # str to int

#SPECIES = ['0','1','2','3','4','5','6','7','8','9']
#SPECIES_INDEX = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}

# Let's try it out...
for name in SPECIES:
    print(f"{name} maps to {SPECIES_INDEX[name]}")


A = df_model1.to_numpy()   
A = A.astype('float64')    # many types:  www.tutorialspoint.com/numpy/numpy_data_types.htm
print(A[0:5,:])


NUM_ROWS, NUM_COLS = A.shape
print(f"\nThe dataset has {NUM_ROWS} rows and {NUM_COLS} cols")


print("+++ Start of data definitions +++\n")

#
# we could do this at the data-frame level, too!
#

X_all = A[:,0:64]   # X (features) ... is all rows, columns 0, 1, 2, 3 thru 63
y_all = A[:,64:]    # y (labels) ... is all rows, columns 64

print(f"y_all (just the labels/species, first few rows) are \n {y_all[0:5]}")
print()
print(f"X_all (just the features, first few rows) are \n {X_all[0:5]}")


indices = np.random.permutation(len(y_all))  # indices is a permutation-list

# we scramble both X and y, necessarily with the same permutation
X_all = X_all[indices]              # we apply the _same_ permutation to each!
y_all = y_all[indices]              # again...
print(f"The scrambled labels/species are \n {y_all[0:5]}")
print()
print(f"The corresponding data rows are \n {X_all[0:5]}")


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
        print(f"{str(X[i,0:4]):>58s} -> {'?':^7s} {str(y[i]):<21s}")   # !s is str ...
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


print(A[1])

print(SPECIES)


def get_species(A):
    """ returns the species for A ~ [1 0 0] or [0 1 0] or ... """
    for i in range(len(SPECIES)):
        if A[i] == 1: 
            return SPECIES[i]  # note that this "takes the first one"
    return "no species" 

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
        pred_probs = str(prediction_probs[i,:])
        desired = y[i].astype(int)
        # print(pred, desired, pred_probs)
        pred_species = get_species(pred)
        des_species  = get_species(desired)
        if pred_species != des_species: result = "  incorrect: " + pred_probs
        else: result = "  correct" + (": "+pred_probs if SEE_PROBS else "") ; num_correct += 1
        # Xpr = Xsc  # if you want to see the scaled versions
        print(f"{Xpr[i,0:4]!s:>28s} -> {pred_species:^12s} {des_species:12s} {result:^10s}") 
    print(f"\ncorrect predictions: {num_correct} out of {len(y)}")
    
#
# let's see how it did on the test data (also the training data!)
#
ascii_table_for_classifier(X_test_scaled,
                           y_test_scaled,
                           nn_classifier,
                           scaler)  


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


########




def get_species(A):
    """ returns the species for A ~ [1 0 0] or [0 1 0] or ... """
    for i in range(len(SPECIES)):
        if A[i] == 1: 
            return SPECIES[i]  # note that this "takes the first one"
    return "no species" 


def predictive_model( Features, MODEL, SCALER ):
    """ input: a list of four features 
                [ sepallen, sepalwid, petallen, petalwid ]
        output: the predicted species of iris, from
                  setosa (0), versicolor (1), virginica (2)
    """
    our_features = np.asarray([Features])                 # extra brackets needed
    scaled_features = SCALER.transform(our_features)      # we have to scale the features into "scaled space"
    predicted_cat = MODEL.predict(scaled_features)        # then, the nnet can predict our "cat" variables
    prediction_probs = nn.predict_proba(scaled_features) # all prediction probabilities
    # our_features = SCALER.inverse_transform(scaled_features)  # we can convert back (optional!)
    predicted_species = get_species(predicted_cat[0])     # (it's extra-nested) get the species name
    return predicted_species, prediction_probs
   
#
# Try it!
# 
# Features = eval(input("Enter new Features: "))
#
Features = [0,0,0,8,14,0,0,0,0,0,5,16,11,0,0,0,0,1,15,14,1,6,0,0,0,7,16,5,3,16,8,0,0,8,16,8,14,16,2,0,0,0,6,14,16,11,0,0,0,0,0,6,16,4,0,0,0,0,0,10,15,0,0,0] # [5.8,2.7,4.1,1.0] # [4.6,3.6,3.0,2.2] [6.7,3.3,5.7,2.1]

LoD = [[0,0,0,8,14,0,0,0,0,0,5,16,11,0,0,0,0,1,15,14,1,6,0,0,0,7,16,5,3,16,8,0,0,8,16,8,14,16,2,0,0,0,6,14,16,11,0,0,0,0,0,6,16,4,0,0,0,0,0,10,15,0,0,0],
[0,0,0,5,14,12,2,0,0,0,7,15,8,14,4,0,0,0,6,2,3,13,1,0,0,0,0,1,13,4,0,0,0,0,1,11,9,0,0,0,0,8,16,13,0,0,0,0,0,5,14,16,11,2,0,0,0,0,0,6,12,13,3,0],
[0,0,0,3,16,3,0,0,0,0,0,12,16,2,0,0,0,0,8,16,16,4,0,0,0,7,16,15,16,12,11,0,0,8,16,16,16,13,3,0,0,0,0,7,14,1,0,0,0,0,0,6,16,0,0,0,0,0,0,4,14,0,0,0],
[0,0,0,3,15,10,1,0,0,0,0,11,10,16,4,0,0,0,0,12,1,15,6,0,0,0,0,3,4,15,4,0,0,0,0,6,15,6,0,0,0,4,15,16,9,0,0,0,0,0,13,16,15,9,3,0,0,0,0,4,9,14,7,0],
[0,0,0,3,16,3,0,0,0,0,0,10,16,11,0,0,0,0,4,16,16,8,0,0,0,2,14,12,16,5,0,0,0,10,16,14,16,16,11,0,0,5,12,13,16,8,3,0,0,0,0,2,15,3,0,0,0,0,0,4,12,0,0,0],
[0,0,7,15,15,4,0,0,0,8,16,16,16,4,0,0,0,8,15,8,16,4,0,0,0,0,0,10,15,0,0,0,0,0,1,15,9,0,0,0,0,0,6,16,2,0,0,0,0,0,8,16,8,11,9,0,0,0,9,16,16,12,3,0]]


SEE_PROBS = False

# run on each one:
for Features in LoD:
    MODEL = nn_classifier
    SCALER = scaler
    name, probs = predictive_model( Features, MODEL, SCALER )  # pass in the model, too!
    prob_str = "   with probs: " + str(probs) if SEE_PROBS == True else ""
    print(f"I predict {name:>12s} from the features {Features}  {prob_str}")    # Answers in the assignment...


#
# coding cells, for copy-paste-and-adapt...

print(COL_INDEX)
print()
print(COLUMNS)


print(range(0,65))



print("+++ Start of data-assembly for feature-regression! +++\n")
# construct the correct X_all from the columns we want
# we use np.concatenate to combine parts of the dataset to get all-except-column 0:
#                     exclude 0  , include 1 to the end

predictedPixelValues = []
errorList = []

for i in range(0,64): # or just for i in [42] for the first iteration of this

    # when i = 0: sepallen, i = 1: sepalwid, i = 2: petallen, i = 3: petalwid

    X_all = np.concatenate( (A[:,0:i], A[:,i+1:]), axis=1)   
    # X_all = np.concatenate( (A[:,0:i], A[:,i+1:]), axis=1)    # if i == 42, we should be excluding that pixel


    y_all = A[:,i]                    # y (labels) ... is all of column 0, sepallen (sepal length) 
    #                                 # change the line above to make other columns the target (y_all)


    # we scramble the data, to give a different TRAIN/TEST split each time...
    # 
    indices = np.random.permutation(len(y_all))  # indices is a permutation-list

    # we scramble both X and y, necessarily with the same permutation
    X_all = X_all[indices]              # we apply the _same_ permutation to each!
    y_all = y_all[indices]              # again...


    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2)

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
    # ascii_table(X_train_scaled[0:5,:],y_train_scaled[0:5],None)

    # reused from above - seeing the unscaled data (inverting the scaler)
    # ascii_table(X_train_scaled[0:5,:],y_train_scaled[0:5],scaler)

    # MLPRegressor predicts _floating-point_ outputs


    from sklearn.neural_network import MLPRegressor

    nn_regressor = MLPRegressor(hidden_layer_sizes=(6,7), 
                        max_iter=500,          # how many training epochs
                        verbose=True,          # do we want to watch as it trains?
                        shuffle=True,          # shuffle each epoch?
                        random_state=None,     # use for reproducibility
                        learning_rate_init=.1, # how much of each error to back-propagate
                        learning_rate = 'adaptive')  # how to handle the learning_rate

    nn_regressor.fit(X_train_scaled, y_train_scaled)


    def ascii_table_for_regressor(Xsc,y,nn,scaler):
        """ a table including predictions using nn.predict """
        predictions = nn.predict(Xsc) # all predictions
        Xpr = scaler.inverse_transform(Xsc)  # Xpr is the "X to print": unscaled data!
        # measure error
        error = 0.0
        # printing
        # print(f"{'input ':>35s} ->  {'pred':^6s}  {'des.':^6s}   {'absdiff':^10s}") 
        for i in range(len(y)):
            pred = predictions[i]
            desired = y[i]
            result = abs(desired - pred)
            error += result
            # Xpr = Xsc   # if you'd like to see the scaled values
            # print(f"{Xpr[i,:]!s:>35s} ->  {pred:<+6.2f}  {desired:<+6.2f}   {result:^10.2f}") 
        # print()
        # print()
        print(f"predicted pixel value for pixel {i}: {pred}")
        predictedPixelValues.append(float(pred))
        errorList.append(float(error/len(y)))

        # print("\n" + "+++++   +++++   +++++           ")
        # print(f"average abs diff error:   {error/len(y):<6.3f}")
        # print("+++++   +++++   +++++           ")






        
    # let's see how it did on the test data (also the training data!)
    #
    ascii_table_for_regressor(X_test_scaled,
                            y_test_scaled,
                            nn_regressor,
                            scaler)   # this is our own f'n, above
    



# result

print(len(predictedPixelValues))

print(predictedPixelValues)

print(predictedPixelValues[41])


# plotting heat map of each pixel prediction (will do error next)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
sns.set_theme()


pixels_as_row = predictedPixelValues
print("pixels as 1d numpy array (row):\n", pixels_as_row)

pixels_as_image = np.reshape(pixels_as_row, (8,8))   # reshape into a 2d 8x8 array (image)
print("\npixels as 2d numpy array (image):\n", pixels_as_image)






# create the figure, f, and the axes, ax:
f, ax = plt.subplots(figsize=(9, 6))

# colormap choice! Fun!   www.practicalpythonfordatascience.com/ap_seaborn_palette or seaborn.pydata.org/tutorial/color_palettes.html
our_colormap = sns.color_palette("light:b", as_cmap=True) 

# Draw a heatmap with the numeric values in each cell (make annot=False to remove the values)
sns.heatmap(pixels_as_image, annot=True, linewidths=.5, ax=ax, cmap=our_colormap)


# now looking at error for each pixel




#plotting error (ie predictability map)

pixels_as_row = errorList
print("pixels as 1d numpy array (row):\n", pixels_as_row)

pixels_as_image = np.reshape(pixels_as_row, (8,8))   # reshape into a 2d 8x8 array (image)
print("\npixels as 2d numpy array (image):\n", pixels_as_image)


# create the figure, f, and the axes, ax:
f, ax = plt.subplots(figsize=(9, 6))

# colormap choice! Fun!   www.practicalpythonfordatascience.com/ap_seaborn_palette or seaborn.pydata.org/tutorial/color_palettes.html
our_colormap = sns.color_palette("light:b", as_cmap=True) 

# Draw a heatmap with the numeric values in each cell (make annot=False to remove the values)
sns.heatmap(pixels_as_image, annot=True, linewidths=.5, ax=ax, cmap=our_colormap)


