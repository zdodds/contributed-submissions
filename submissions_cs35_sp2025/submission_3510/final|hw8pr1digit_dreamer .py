#
# Here, we have a one-pixel predictor, to get you started...



# libraries!
import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)
import seaborn as sns
import matplotlib.pyplot as plt


# let's read in our digits data...
# 
# for read_csv, use header=0 when row 0 is a header row
# 
filename = 'digits.csv'
df = pd.read_csv(filename, header=0)   # encoding="utf-8" et al.
print(f"{filename} : file read into a pandas dataframe.")



#
# let's drop that last column (dropping is usually by _name_):
#
#   if you want a list of the column names use df.columns
coltodrop = df.columns[65]     # get last column name (with the url)
df_clean = df.drop(columns=[coltodrop])  # drop by name is typical
df_clean.info()                         # should be happier!



#
# let's keep our column names in variables, for reference
#
COLUMNS = df_clean.columns            # "list" of columns
print(f"COLUMNS: {COLUMNS}")  

# let's create a dictionary to look up any column index by name
COL_INDEX = {}
for i, name in enumerate(COLUMNS):
    COL_INDEX[name] = i  # using the name (as key), look up the value (i)
print(f"COL_INDEX: {COL_INDEX}")

# and for our "SPECIES"!
SPECIES = [ str(i) for i in range(0,10) ]  # list with a string at each index (index -> string)
SPECIES_INDEX = { s:int(s) for s in SPECIES }  # dictionary mapping from string -> index

# and our "target labels"
print(f"SPECIES: {SPECIES}")  
print(f"SPECIES_INDEX: {SPECIES_INDEX}")


#
# let's convert our dataframe to a numpy array, named A
#    Our ML library, scikit-learn operates entirely on numpy arrays.
#
A = df_clean.to_numpy()    # .values gets the numpy array
A = A.astype('float64')  # so many:  www.tutorialspoint.com/numpy/numpy_data_types.htm
print(f"A's shape is {A.shape}")
print(A)


#
# You will explore a different direction: "hallucinating" new data!
#      This is sometimes called "imputing" missing data.
#

# First, build a regressor that
#      + uses the first 48 pixels (6 image rows) to predict the floating-point value of pix52
#      + we'll see how accurate it is...
#      + then, you'll expand this process to build a regressor for _each_ pixel indexed from 48-63
#      + and use those to "imagine" the bottom two rows of the digits...


#
# some starting code is provided here...
#


#
# regression model that uses as input the first 48 pixels (pix0 to pix47)
#                       and, as output, predicts the value of pix52
#

print("+++ Start of regression prediction of pix52! +++\n")

X_all = A[:,0:48]  ### old: np.concatenate( (A[:,0:3], A[:,4:]),axis=1)  # horizontal concatenation
y_all = A[:,52]    # y (labels) ... is all rows, column indexed 52 (pix52) only (actually the 53rd pixel, but ok)

print(f"y_all (just target values, pix52)   is \n {y_all}") 
print(f"X_all (just features: 3 rows) is \n {X_all[:3,:]}")


#
# we scramble the data, to give a different TRAIN/TEST split each time...
# 
indices = np.random.permutation(len(y_all))  # indices is a permutation-list

# we scramble both X and y, necessarily with the same permutation
X_all = X_all[indices]              # we apply the _same_ permutation to each!
y_all = y_all[indices]              # again...
print("labels (target)\n",y_all)
print("features\n", X_all[:3,:])


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

# Here is a fully-scaled dataset:

X_all_scaled = scaler.transform(X_all)
y_all_scaled = y_all.copy()      # not scaled


# Here are our scaled training and testing sets:

X_train_scaled = scaler.transform(X_train) # scale!
X_test_scaled = scaler.transform(X_test) # scale!

y_train_scaled = y_train  # the predicted/desired labels are not scaled
y_test_scaled = y_test  # not using the scaler

def ascii_table(X,y):
    """ print a table of binary inputs and outputs """
    print(f"{'input ':>70s} -> {'pred':<5s} {'des.':<5s}") 
    for i in range(len(y)):
        s_to_show = str(X[i,:])
        s_to_show = s_to_show[0:60]
        print(f"{s_to_show!s:>70s} -> {'?':<5s} {y[i]:<5.0f}")   # !s is str ...
    
ascii_table(X_train_scaled[0:5,:],y_train_scaled[0:5])

#
# Note that the zeros have become -1's
# and the 1's have stayed 1's
#


#
# MLPRegressor predicts _floating-point_ outputs
#

from sklearn.neural_network import MLPRegressor

nn_regressor = MLPRegressor(hidden_layer_sizes=(6,7), 
                    max_iter=342,          # how many training epochs
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
# how did it do? now we're making progress (by regressing)
#

def ascii_table_for_regressor(Xsc,y,nn,scaler):
    """ a table including predictions using nn.predict """
    predictions = nn.predict(Xsc) # all predictions
    Xpr = scaler.inverse_transform(Xsc)  # Xpr is the "X to print": unscaled data!
    # measure error
    error = 0.0
    # printing
    print(f"{'input ':>35s} ->  {'pred':^6s}  {'des.':^6s}  {'absdiff':^10s}") 
    for i in range(len(y)):
        pred = predictions[i]
        desired = y[i]
        result = abs(desired - pred)
        error += result
        # Xpr = Xsc   # if you'd like to see the scaled values
        s_to_show = str(Xpr[i,:])
        s_to_show = s_to_show[0:25]  # we'll just take 25 of these
        print(f"{s_to_show!s:>35s} ->  {pred:<+6.3f}  {desired:<+6.3f}  {result:^10.3f}") 

    print("\n" + "+++++   +++++      +++++   +++++   ")
    print(f"average abs error: {error/len(y)}")
    print("+++++   +++++      +++++   +++++   ")
    
#
# let's see how it did on the test data 
# 
if True:
    ascii_table_for_regressor(X_test_scaled,
                            y_test_scaled,
                            nn_regressor,
                            scaler)   # this is our own f'n, above

# and how it did on the training data!
#
if False:
    ascii_table_for_regressor(X_train_scaled,
                            y_train_scaled,
                            nn_regressor,
                            scaler)   # this is our own f'n, above



#
# let's create a final nn_regressor for pix52
#
pix52_final_regressor = MLPRegressor(hidden_layer_sizes=(6,7), 
                                    max_iter=400, 
                                    activation="tanh",
                                    solver='sgd', 
                                    verbose=False, 
                                    shuffle=True,
                                    random_state=None, # reproduceability!
                                    learning_rate_init=.1, 
                                    learning_rate = 'adaptive')

print("\n\n++++++++++  TRAINING:  begin  +++++++++++++++\n\n")
pix52_final_regressor.fit(X_all_scaled, y_all_scaled)
print("\n\n++++++++++  TRAINING:   end  +++++++++++++++\n\n")

print(f"The (sq) prediction error (the loss) is {pix52_final_regressor.loss_}") 
print(f"So, the 'average' error per pixel is {pix52_final_regressor.loss_**0.5}")


#
# and, let's be sure we can use our "finalized" model:
#

def predict_from_model(pixels, model):
    """ returns the prediction on the input pixels using the input model
    """
    pixels_array = np.asarray([pixels])   # the extra sq. brackets are needed!
    pixels_scaled = scaler.transform(pixels_array)  # need to use the scaler!
    predicted = model.predict(pixels_scaled)
    return predicted

#
# let's choose a digit to try...
#
row_to_show = 4                         # different indexing from X_all and y_all (they were reordered)
numeral = A[row_to_show,64]
print(f"The numeral is a {int(numeral)}\n")

all_pixels = A[row_to_show,0:64] 
first48pixels = A[row_to_show,0:48] 

pix52_predicted = predict_from_model(first48pixels,pix52_final_regressor)
pix52_actual = A[row_to_show,52]

print(f"pix52 [predicted] vs. actual:  {pix52_predicted} vs. {pix52_actual}")


#
# Let's visualize!   Here's the idea: 
# 
# Choose a row index (row_to_show)
# Show the original digit
# Show the original digit with pix52 replaced (may not be noticeable...)
# show the original digit with the bottom-two rows zero'ed out _except_ pix 52 :-)
#


#
# Let's create a function to show one digit
#

def show_digit( pixels ):
    """ should create a heatmap (image) of the digit contained in row 
            input: pixels should be a 1d numpy array
            if it's more then 64 values, it will be truncated
            if it's fewer than 64 values, 0's will be appended
            
    """
    # make sure the sizes are ok!
    num_pixels = len(pixels)
    if num_pixels != 64:
        print(f"(in show_digit) num_pixels was {num_pixels}; now set to 64")
    if num_pixels > 64:   # an elif would be a poor choice here, as I found!
        pixels = pixels[0:64]
    if num_pixels < 64:   
        num_zeros = 64-len(pixels)
        pixels = np.concatenate( (pixels, np.zeros(num_zeros)), axis=0 )
        
    pixels = pixels.astype(int)         # convert to integers for plotting
    pixels = np.reshape(pixels, (8,8))  # make 8x8
    # print(f"The pixels are\n{pixels}")  
    f, ax = plt.subplots(figsize=(9, 6))  # Draw a heatmap w/option of numeric values in each cell
    
    #my_cmap = sns.dark_palette("Purple", as_cmap=True)
    my_cmap = sns.light_palette("Gray", as_cmap=True)    # all seaborn palettes: medium.com/@morganjonesartist/color-guide-to-seaborn-palettes-da849406d44f
    # plot! annot=True to see the values...   palettes listed at very bottom of this notebook
    sns.heatmap(pixels, annot=False, fmt="d", linewidths=.5, ax=ax, cmap=my_cmap) # 'seismic'


#
# Another example of predicting one pixel
#
row_to_show = 42                         
numeral = A[row_to_show,64]
print(f"The numeral is a {int(numeral)}\n")
# show all from the original data
show_digit( A[row_to_show,0:64] )   # show full original

all_pixels = A[row_to_show,0:64].copy()
first48pixels = all_pixels[0:48] 

pix52_predicted = predict_from_model(first48pixels,pix52_final_regressor)
pix52_actual = A[row_to_show,52]

print(f"pix52 [predicted] vs. actual:  {pix52_predicted} {pix52_actual}")

# erase last 16 pixels
all_pixels[48:64] = np.zeros(16)

# show without pix52
all_pixels[52] = 0         # omit this one
show_digit( all_pixels )   # show without pixel 52

# show with pix52
all_pixels[52] = np.round(pix52_predicted)    # include this one
show_digit( all_pixels )   # show with pixel 52





#
# Adapt from the previous example:
#


DoR = {}
XaSD = {}
YaSD = {}

for i in range(48,64):
    X_all2 = A[:, 0:48]
    y_all2 = A[:, i] 

    #print(f"y_all (just target values, pix" + str(i) +  " is \n {y_all}") 
    #print(f"X_all (just features: 3 rows) is \n {X_all[:3,:]}")

    from sklearn.model_selection import train_test_split
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_all2, y_all2, test_size=0.2)

    X_train_scaled2 = scaler.transform(X_train2) # scale!
    X_test_scaled2 = scaler.transform(X_test2) # scale!

    y_train_scaled2 = y_train2  # the predicted/desired labels are not scaled
    y_test_scaled2 = y_test2  # not using the scaler

    X_all_scaled2 = scaler.transform(X_all2)
    y_all_scaled2 = y_all2.copy() 

    XaSD[i] = X_all_scaled2
    YaSD[i] = y_all_scaled2

    
    #ascii_table(X_train_scaled[0:5,:],y_train_scaled[0:5])
    #from sklearn.neural_network import MLPRegressor
    
    nn_regressor = MLPRegressor(hidden_layer_sizes=(6,7), 
                    max_iter=342,          # how many training epochs
                    activation="tanh",     # the activation function
                    solver='sgd',          # the optimizer
                    verbose=True,          # do we want to watch as it trains?
                    shuffle=True,          # shuffle each epoch?
                    random_state=None,     # use for reproducibility
                    learning_rate_init=.1, # how much of each error to back-propagate
                    learning_rate = 'adaptive')  # how to handle the learning_rate
    #print("\n\n++++++++++  TRAINING:  begin  +++++++++++++++\n\n")
    nn_regressor.fit(X_train_scaled2, y_train_scaled2)
    #print("++++++++++  TRAINING:   end  +++++++++++++++")

    DoR[ i ] =  nn_regressor 

    if True:
        ascii_table_for_regressor(X_test_scaled2,
                            y_test_scaled2,
                            DoR[ i ],
                            scaler)   # this is our own f'n, above
    # and how it did on the training data!
    # 
    if False:
        ascii_table_for_regressor(X_train_scaled2,
                            y_train_scaled2,
                            DoR[ i ],
                            scaler)   # this is our own f'n, above

    #print("\n\n++++++++++  TRAINING:  begin  +++++++++++++++\n\n")
    #nn_regressor.fit(X_train_scaled2, y_train_scaled2)
    #print("++++++++++  TRAINING:   end  +++++++++++++++")

    #print(f"The (squared) prediction error (the loss) is {nn_regressor.loss_}")
    #print(f"And, its square root: {nn_regressor.loss_ ** 0.5}")


   # if True:
        ascii_table_for_regressor(X_test_scaled2,
                            y_test_scaled2,
                            DoR[ i ],
                            scaler)   # this is our own f'n, above
        # and how it did on the training data!
        # #
    #if False:
        ascii_table_for_regressor(X_train_scaled2,
                            y_train_scaled2,
                            DoR[ i ],
                            scaler)   # this is our own f'n, above

    

    


# a final nn_regressor for pix53

pix53_final_regressor = MLPRegressor(hidden_layer_sizes=(6,7), 
                                    max_iter=400, 
                                    activation="tanh",
                                    solver='sgd', 
                                    verbose=False, 
                                    shuffle=True,
                                    random_state=None, # reproduceability!
                                    learning_rate_init=.1, 
                                    learning_rate = 'adaptive')

print("\n\n++++++++++  TRAINING:  begin  +++++++++++++++\n\n")
pix53_final_regressor.fit(XaSD[53], YaSD[53])
print("\n\n++++++++++  TRAINING:   end  +++++++++++++++\n\n")

print(f"The (sq) prediction error (the loss) is {pix53_final_regressor.loss_}") 
print(f"So, the 'average' error per pixel is {pix53_final_regressor.loss_**0.5}")


pix53_predicted = predict_from_model(first48pixels,pix53_final_regressor)
pix53_actual = A[row_to_show,53]
print(f"pix53 [predicted] vs. actual:  {pix53_predicted} vs. {pix53_actual}")


row_to_show = 50                         
numeral = A[row_to_show,64]
print(f"The numeral is a {int(numeral)}\n")
# show all from the original data
show_digit( A[row_to_show,0:64] )   # show full original

all_pixels = A[row_to_show,0:64].copy()
first48pixels = all_pixels[0:48] 

pix53_predicted = predict_from_model(first48pixels,pix53_final_regressor)
pix53_actual = A[row_to_show,53]

print(f"pix53 [predicted] vs. actual:  {pix53_predicted} {pix53_actual}")

# erase last 16 pixels
all_pixels[48:64] = np.zeros(16)
# show without pix53
all_pixels[53] = 0         # omit this one
show_digit( all_pixels )   # show without pixel 52

# show with pix53
all_pixels[53] = np.round(pix53_predicted)    # include this one
show_digit( all_pixels )   # show with pixel 52


pix54_final_regressor = MLPRegressor(hidden_layer_sizes=(6,7), 
                                    max_iter=400, 
                                    activation="tanh",
                                    solver='sgd', 
                                    verbose=False, 
                                    shuffle=True,
                                    random_state=None, # reproduceability!
                                    learning_rate_init=.1, 
                                    learning_rate = 'adaptive')
pix54_final_regressor.fit(XaSD[54], YaSD[54])

print(f"The (sq) prediction error (the loss) is {pix54_final_regressor.loss_}") 
print(f"So, the 'average' error per pixel is {pix54_final_regressor.loss_**0.5}")


pix54_predicted = predict_from_model(first48pixels,pix54_final_regressor)
pix54_actual = A[row_to_show,54]
print(f"pix53 [predicted] vs. actual:  {pix54_predicted} vs. {pix54_actual}")


row_to_show = 64                         
numeral = A[row_to_show,64]
print(f"The numeral is a {int(numeral)}\n")
# show all from the original data
show_digit( A[row_to_show,0:64] )   # show full original

all_pixels = A[row_to_show,0:64].copy()
first48pixels = all_pixels[0:48] 

pix54_predicted = predict_from_model(first48pixels,pix54_final_regressor)
pix54_actual = A[row_to_show,54]

print(f"pix54 [predicted] vs. actual:  {pix54_predicted} {pix54_actual}")

# erase last 16 pixels
all_pixels[48:64] = np.zeros(16)
# show without pix53
all_pixels[54] = 0         # omit this one
show_digit( all_pixels )   # show without pixel 52

# show with pix53
all_pixels[54] = np.round(pix54_predicted)    # include this one
show_digit( all_pixels )   # show with pixel 52


pix55_final_regressor = MLPRegressor(hidden_layer_sizes=(6,7), 
                                    max_iter=400, 
                                    activation="tanh",
                                    solver='sgd', 
                                    verbose=False, 
                                    shuffle=True,
                                    random_state=None, # reproduceability!
                                    learning_rate_init=.1, 
                                    learning_rate = 'adaptive')
pix55_final_regressor.fit(XaSD[55], YaSD[55])

print(f"The (sq) prediction error (the loss) is {pix55_final_regressor.loss_}") 
print(f"So, the 'average' error per pixel is {pix55_final_regressor.loss_**0.5}")


pix55_predicted = predict_from_model(first48pixels,pix55_final_regressor)
pix55_actual = A[row_to_show,55]
print(f"pix55 [predicted] vs. actual:  {pix55_predicted} vs. {pix55_actual}")


row_to_show = 64                         
numeral = A[row_to_show,64]
print(f"The numeral is a {int(numeral)}\n")
# show all from the original data
show_digit( A[row_to_show,0:64] )   # show full original

all_pixels = A[row_to_show,0:64].copy()
first48pixels = all_pixels[0:48] 

pix55_predicted = predict_from_model(first48pixels,pix55_final_regressor)
pix55_actual = A[row_to_show,55]

print(f"pix55 [predicted] vs. actual:  {pix55_predicted} {pix55_actual}")

# erase last 16 pixels
all_pixels[48:64] = np.zeros(16)
# show without pix53
all_pixels[55] = 0         # omit this one
show_digit( all_pixels )   # show without pixel 52

# show with pix53
all_pixels[55] = np.round(pix54_predicted)    # include this one
show_digit( all_pixels )   # show with pixel 52


DoR2 = {}
XaSD2 = {}
YaSD2 = {}

for i in range(32,64):
    X_all3 = A[:, 0:32]
    y_all3 = A[:, i] 

    #print(f"y_all (just target values, pix" + str(i) +  " is \n {y_all}") 
    #print(f"X_all (just features: 3 rows) is \n {X_all[:3,:]}")

    from sklearn.model_selection import train_test_split
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X_all3, y_all3, test_size=0.2)

    scaler32 = StandardScaler()
    scaler32.fit(X_train3)  # Sc

    X_train_scaled3 = scaler32.transform(X_train3) # scale!
    X_test_scaled3 = scaler32.transform(X_test3) # scale!

    y_train_scaled3 = y_train3  # the predicted/desired labels are not scaled
    y_test_scaled3 = y_test3  # not using the scaler

    X_all_scaled3 = scaler32.transform(X_all3)
    y_all_scaled3 = y_all3.copy() 

    XaSD2[i] = X_all_scaled3
    YaSD2[i] = y_all_scaled3

    
    #ascii_table(X_train_scaled[0:5,:],y_train_scaled[0:5])
    #from sklearn.neural_network import MLPRegressor
    
    nn_regressor = MLPRegressor(hidden_layer_sizes=(6,7), 
                    max_iter=342,          # how many training epochs
                    activation="tanh",     # the activation function
                    solver='sgd',          # the optimizer
                    verbose=True,          # do we want to watch as it trains?
                    shuffle=True,          # shuffle each epoch?
                    random_state=None,     # use for reproducibility
                    learning_rate_init=.1, # how much of each error to back-propagate
                    learning_rate = 'adaptive')  # how to handle the learning_rate
    #print("\n\n++++++++++  TRAINING:  begin  +++++++++++++++\n\n")
    nn_regressor.fit(X_train_scaled3, y_train_scaled3)
    #print("++++++++++  TRAINING:   end  +++++++++++++++")

    DoR2[ i ] =  nn_regressor 

    if True:
        ascii_table_for_regressor(X_test_scaled3,
                            y_test_scaled3,
                            DoR2[ i ],
                            scaler32)   # this is our own f'n, above
    # and how it did on the training data!
    # 
    if False:
        ascii_table_for_regressor(X_train_scaled2,
                            y_train_scaled2,
                            DoR[ i ],
                            scaler)   # this is our own f'n, above


final_regressor = MLPRegressor(hidden_layer_sizes=(6,7), 
                                    max_iter=400, 
                                    activation="tanh",
                                    solver='sgd', 
                                    verbose=False, 
                                    shuffle=True,
                                    random_state=None, # reproduceability!
                                    learning_rate_init=.1, 
                                    learning_rate = 'adaptive')

for j in range(32,64):
    #print("\n\n++++++++++  TRAINING:  begin  +++++++++++++++\n\n")
    final_regressor.fit(XaSD2[j],YaSD2[j])
    #print("\n\n++++++++++  TRAINING:   end  +++++++++++++++\n\n")
    
    #print(f"The (sq) prediction error (the loss) is {final_regressor.loss_}") 
    #print(f"So, the 'average' error per pixel is {final_regressor.loss_**0.5}")
    pix_predicted = predict_from_model(first32pixels,final_regressor)
    pix_actual = A[row_to_show,j]
    print(f"pix{j} [predicted] vs. actual: {pix_predicted} vs. {pix_actual}")



import numpy as np

def predict_from_model(input_pixels, model):
    """
    Ensure the input is a 2D array of shape (1, n_features),
    then call the model's predict method.
    """
    # Convert to numpy array (if not already) and force it to 2D.
    input_pixels = np.array(input_pixels)
    if input_pixels.ndim == 1:
        input_pixels = input_pixels.reshape(1, -1)
    # Now predict (the output is assumed to be an array with one element)
    return model.predict(input_pixels)

# ----- Start of main code ----- #

# Suppose A is your data array where each row is a flattened 64-pixel image.
# We assume column 64 holds the numeral label.
row_to_show = 64  
original_image = A[row_to_show, 0:64].copy()  # shape: (64,)

# Display the complete original image for reference.
print(f"The numeral is a {int(A[row_to_show, 64])}\n")
show_digit(original_image)

# Get the top half of the image (first 32 pixels).
# We'll keep this as a 1D array for later concatenation.
observed_top_pixels = original_image[0:32].copy()  # shape: (32,)

# Prepare a 2D version of the top pixels for passing to the model.
observed_top_pixels_2d = observed_top_pixels.reshape(1, -1)  # shape: (1, 32)

predicted_bottom_pixels = []  # To store predicted values for pixels 32 through 63.

# Loop through each pixel index of the missing (bottom four rows) portion.
for j in range(32, 64):
    # Train the regressor for pixel j.
    final_regressor.fit(XaSD2[j], YaSD2[j])
    
    # Predict for pixel j using the 2D version of top pixels.
    pixel_prediction = predict_from_model(observed_top_pixels_2d, final_regressor)
    
    # If the prediction returns an array (even with one element), extract the scalar.
    predicted_value = np.round(pixel_prediction[0])
    
    predicted_bottom_pixels.append(predicted_value)
    print(f"pix{j} [predicted] vs. actual: {predicted_value} vs. {original_image[j]}")

# Convert the list of predicted bottom pixels into a 1D numpy array of length 32.
predicted_bottom_pixels = np.array(predicted_bottom_pixels).flatten()

# Reconstruct the full predicted image by concatenating the observed top half with the predicted bottom half.
# Both parts are 1D (length 32 each), so the result is a 1D array with 64 elements.
predicted_image = np.concatenate([observed_top_pixels, predicted_bottom_pixels])

# Optionally, if your display function expects an 8x8 array, you can reshape it:
# predicted_image = predicted_image.reshape(8, 8)
show_digit(predicted_image)



# Suppose A is your data array, and row_to_show is the index of the image you want to work with.
# Note that we assume each image is represented as a flat vector of 64 pixels.
row_to_show = 64  
original_image = A[row_to_show, 0:64].copy()

# Display the original complete image.
# (Assuming that A[row_to_show,64] holds the numeral label; adjust if needed.)
print(f"The numeral is a {int(A[row_to_show, 64])}\n")
show_digit(original_image)  # Show the full original image

top_pixels = original_image[0:32]  
#top_pixels = top_pixels.reshape(1, -1)
predicted_bottom_pixels = []

# Loop through each pixel index of the missing (bottom four rows) portion.
for j in range(32, 64):
    final_regressor.fit(XaSD2[j], YaSD2[j])
    pixel_prediction = predict_from_model(top_pixels, final_regressor)
    predicted_bottom_pixels.append((np.round(pixel_prediction)))
    #print(f"pix{j} [predicted] vs. actual: {pixel_prediction} vs. {original_image[j]}")
print(np.array(predicted_bottom_pixels).flatten())

# Combine the unchanged top part with the predicted bottom part to form a full predicted image.
predicted_image = np.concatenate([top_pixels, np.array(predicted_bottom_pixels).flatten()])

# Display the image with the predicted bottom four rows.
show_digit(predicted_image)



# Suppose A is your data array, and row_to_show is the index of the image you want to work with.
# Note that we assume each image is represented as a flat vector of 64 pixels.
row_to_show = 66  
original_image = A[row_to_show, 0:64].copy()

# Display the original complete image.
# (Assuming that A[row_to_show,64] holds the numeral label; adjust if needed.)
print(f"The numeral is a {int(A[row_to_show, 64])}\n")
show_digit(original_image)  # Show the full original image

top_pixels = original_image[0:32]  
#top_pixels = top_pixels.reshape(1, -1)
predicted_bottom_pixels = []

# Loop through each pixel index of the missing (bottom four rows) portion.
for j in range(32, 64):
    final_regressor.fit(XaSD2[j], YaSD2[j])
    pixel_prediction = predict_from_model(top_pixels, final_regressor)
    predicted_bottom_pixels.append((np.round(pixel_prediction)))
    #print(f"pix{j} [predicted] vs. actual: {pixel_prediction} vs. {original_image[j]}")
#print(np.array(predicted_bottom_pixels).flatten())

# Combine the unchanged top part with the predicted bottom part to form a full predicted image.
predicted_image = np.concatenate([top_pixels, np.array(predicted_bottom_pixels).flatten()])

# Display the image with the predicted bottom four rows.
show_digit(predicted_image)



# Suppose A is your data array, and row_to_show is the index of the image you want to work with.
# Note that we assume each image is represented as a flat vector of 64 pixels.
row_to_show = 70  
original_image = A[row_to_show, 0:64].copy()

# Display the original complete image.
# (Assuming that A[row_to_show,64] holds the numeral label; adjust if needed.)
print(f"The numeral is a {int(A[row_to_show, 64])}\n")
show_digit(original_image)  # Show the full original image

top_pixels = original_image[0:32]  
#top_pixels = top_pixels.reshape(1, -1)
predicted_bottom_pixels = []

# Loop through each pixel index of the missing (bottom four rows) portion.
for j in range(32, 64):
    final_regressor.fit(XaSD2[j], YaSD2[j])
    pixel_prediction = predict_from_model(top_pixels, final_regressor)
    predicted_bottom_pixels.append((np.round(pixel_prediction)))
    #print(f"pix{j} [predicted] vs. actual: {pixel_prediction} vs. {original_image[j]}")
#print(np.array(predicted_bottom_pixels).flatten())

# Combine the unchanged top part with the predicted bottom part to form a full predicted image.
predicted_image = np.concatenate([top_pixels, np.array(predicted_bottom_pixels).flatten()])

# Display the image with the predicted bottom four rows.
show_digit(predicted_image)



"""
More details:     Your task is to expand this process, so that...

[1]
  + Your system can predict the value of _any_ of the last 16 pixels from the first 48
  +     Make sure the pixels-used and pixels-predicted are easily changed,
  +     Because this problem also asks you to predict the last 32 pixels (from the first 32...)
[2]
  + Next, predict the value of _any_ of the last 32 pixels from the first 32
  +     This will be smooth if the earlier step is modular + easily changeable
[3]
  + Create "dreamed-digit" images for four digits (your choice)
  +     Use the visualization cells above and below as starting points
  +     Make sure two out of the four use 48 pixels and predict 16
  +     Make sure two out of the four use 32 pixels and predict 32
[4a]
  + Extra!  Read in the file 'partial_digits.csv'
  +     there are 10 digits with _only_ the first six rows (48 pixels)   [the last 16 are artificially set to 0]
  +     there are 10 digits with _only_ the first four rows (32 pixels)  [the last 32 are artificially set to 0]
  +     And, hallucinate the missing data! (just as above)  Visualize!
[4b]
  + Extra!  Then, _Classify_ those newly-imputed digits, using the "dreamt pixels"
  +     (Remember you created a classification network in pr2.)
  +     How does it do?
  +     Compare with how it does if you leave the zeros in the data...


Steps [1]-[3] is an example of "imputing" missing data, and then the EC uses this hallucinated (imputed) data 
so that the original digit-classifier would work.

_Any_ modeling technique can be used to impute missing data. It can be complex (NNet or RF)
or very simple, e.g., replace missing data with the average value of that feature in the dataset. 

In this spirit, check out Google's "Deep Dream" and its "Inceptionism" gallery, e.g.,
https://photos.google.com/share/AF1QipPX0SCl7OzWilt9LnuQliattX4OUCj_8EP65_cTVnBmS1jnYgsGQAieQUc1VQWdgQ?pli=1&key=aVBxWjhwSzg2RjJWLWRuVFBBZEN1d205bUdEMnhB

in which the effects (the weights) learned by the network are allowed to "spill out" into other images.
This is different than the _generated-image_ artifacts (now familiar) ...

Here, it's the weights themselves that are _intentionally_ visualized!
"""


