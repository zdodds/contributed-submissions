import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)
import matplotlib.pyplot as plt


filename = 'digits_cleaned.csv'
df_tidy = pd.read_csv(filename)      # encoding = "utf-8", "latin1"
print(f"{filename} : file read into a pandas dataframe.")


print(f"df_tidy.shape is {df_tidy.shape}\n")
df_tidy.info()
df_tidy.head()


df_tidy_cat = pd.get_dummies(data=df_tidy, prefix="is", columns=['actual_digit'])
print(df_tidy_cat.head())


df_model1 = df_tidy_cat.copy()
df_model1.head()


COLUMNS = df_model1.columns          
print(f"COLUMNS is {COLUMNS}\n")  


COL_INDEX = {}
for i, name in enumerate(COLUMNS):
    COL_INDEX[name] = i  
print(f"COL_INDEX is {COL_INDEX}\n\n")


DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']   # int to str
DIGITS_INDEX = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9}  # str to int



for name in DIGITS:
    print(f"Digit {name} maps to {DIGITS_INDEX[name]}")


A = df_model1.to_numpy()   
A = A.astype('float64')
print("\nFirst few rows of array A:")
print(A[:5,:])


NUM_ROWS, NUM_COLS = A.shape
print(f"\nThe dataset has {NUM_ROWS} rows and {NUM_COLS} cols")


n = 42  
print(f"digit #{n} is {A[n]}")


pixel_cols = [col for col in COL_INDEX.keys() if col.startswith('pix')]
label_cols = [col for col in COL_INDEX.keys() if col.startswith('is_')]


pixel_indices = [COL_INDEX[col] for col in pixel_cols]
label_indices = [COL_INDEX[col] for col in label_cols]


X_all = A[:, pixel_indices]   
y_all = A[:, label_indices]   

print(f"y_all (just the labels/digits, first few rows) are \n {y_all[0:5]}")
print()
print(f"X_all (just the features, first few rows) are \n {X_all[0:5]}")


indices = np.random.permutation(len(y_all))  # indices is a permutation-list
X_all = X_all[indices]              # apply the same permutation to each
y_all = y_all[indices]              # again...

print(f"The scrambled labels/digits are \n {y_all[0:5]}")
print()
print(f"The corresponding data rows are \n {X_all[0:5]}")



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2)

print(f"training with {len(y_train)} rows;  testing with {len(y_test)} rows\n")
print(f"+++ Testing +++   Held-out data... (testing data: {len(y_test)})\n")
print(f"y_test: {y_test[0:5,:]}\n")
print(f"X_test (few rows): {X_test[0:5,:]}")
print("\n")
print(f"+++ Training +++   Data used for modeling... (training data: {len(y_train)})\n")
print(f"y_train: {y_train[0:5,:]}\n")
print(f"X_train (few rows): {X_train[0:5,:]}")


from sklearn.preprocessing import StandardScaler

USE_SCALER = True

if USE_SCALER == True:
    scaler = StandardScaler()
    scaler.fit(X_train)
else:
    scaler = StandardScaler(copy=True, with_mean=False, with_std=False)
    scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_scaled = y_train.copy()
y_test_scaled = y_test.copy()



def ascii_table(X, y, scaler_to_invert=None):
    """ print a table of inputs and outputs """
    np.set_printoptions(precision=2)  # Let's use less precision
    if scaler_to_invert == None:  # don't use the scaler
        X = X
    else:
        X = scaler_to_invert.inverse_transform(X)
    print(f"{'input ':>58s} -> {'pred':^7s} {'des':<5s}")
    for i in range(min(5, len(y))):  # Show only first 5 to avoid large output
        print(f"{str(X[i,0:5]):>58s} -> {'?':^7s} {str(y[i][0:3]):<21s}")
    print()


ascii_table(X_train_scaled[0:5,:], y_train_scaled[0:5], None)
ascii_table(X_train_scaled[0:5,:], y_train_scaled[0:5], scaler_to_invert=scaler)


from sklearn.neural_network import MLPClassifier


nn_classifier = MLPClassifier(hidden_layer_sizes=(32, 16),  
                    max_iter=500,
                    verbose=True,
                    shuffle=True,
                    random_state=42,
                    learning_rate_init=.05,
                    learning_rate='adaptive')



print("\n\n++++++++++  TRAINING:  begin  +++++++++++++++\n\n")
nn_classifier.fit(X_train_scaled, y_train_scaled)
print("\n++++++++++  TRAINING:   end  +++++++++++++++")
print(f"The analog prediction error (the loss) is {nn_classifier.loss_}")


def get_digit(A):
    """ returns the digit for A ~ [1 0 0 0 0 0 0 0 0 0] or [0 1 0 0 0 0 0 0 0 0] or ... """
    for i in range(len(DIGITS)):
        if A[i] == 1: 
            return DIGITS[i]  
    return "no digit" 


def ascii_table_for_classifier(Xsc, y, nn, scaler):
    """ a table including predictions using nn.predict """
    predictions = nn.predict(Xsc)            # all predictions
    prediction_probs = nn.predict_proba(Xsc) # all prediction probabilities
    Xpr = scaler.inverse_transform(Xsc)      # Xpr is the "X to print": the unscaled data
    # count correct
    num_correct = 0
    # printing
    print(f"{'input ':>28s} -> {'pred':^12s} {'des.':^12s}")
    for i in range(min(20, len(y))):  # Show only first 20 to avoid large output
        pred = predictions[i]
        desired = y[i].astype(int)
        pred_digit = get_digit(pred)
        des_digit = get_digit(desired)
        if pred_digit != des_digit: 
            result = "  incorrect"
        else: 
            result = "  correct" 
            num_correct += 1
        print(f"{i:>3}: [...] -> {pred_digit:^12s} {des_digit:12s} {result:^10s}")
    print(f"\ncorrect predictions: {num_correct} out of {min(20, len(y))}")


# Function to visualize the testing results
def ascii_table_for_classifier(Xsc, y, nn, scaler):
    """ a table including predictions using nn.predict """
    predictions = nn.predict(Xsc)            # all predictions
    prediction_probs = nn.predict_proba(Xsc) # all prediction probabilities
    Xpr = scaler.inverse_transform(Xsc)      # Xpr is the "X to print": the unscaled data
    # count correct
    num_correct = 0
    # printing
    print(f"{'input ':>28s} -> {'pred':^12s} {'des.':^12s}")
    for i in range(min(20, len(y))):  # Show only first 20 to avoid large output
        pred = predictions[i]
        desired = y[i].astype(int)
        pred_digit = get_digit(pred)
        des_digit = get_digit(desired)
        if pred_digit != des_digit: 
            result = "  incorrect"
        else: 
            result = "  correct" 
            num_correct += 1
        print(f"{i:>3}: [...] -> {pred_digit:^12s} {des_digit:12s} {result:^10s}")
    print(f"\ncorrect predictions: {num_correct} out of {min(20, len(y))}")
    
    # Calculate overall accuracy
    all_preds = nn.predict(Xsc)
    all_correct = 0
    for i in range(len(y)):
        pred = all_preds[i]
        desired = y[i].astype(int)
        if get_digit(pred) == get_digit(desired):
            all_correct += 1
    print(f"Overall accuracy: {all_correct/len(y)*100:.2f}%")



print("\nTesting results:")
ascii_table_for_classifier(X_test_scaled, y_test_scaled, nn_classifier, scaler)


def predictive_model(Features, MODEL, SCALER):
    """ input: a list of 64 pixel features
        output: the predicted digit, from 0-9
    """
    our_features = np.asarray([Features])                 # extra brackets needed
    scaled_features = SCALER.transform(our_features)      # we have to scale the features into "scaled space"
    predicted_cat = MODEL.predict(scaled_features)        # then, the nnet can predict our "cat" variables
    prediction_probs = MODEL.predict_proba(scaled_features) # all prediction probabilities
    predicted_digit = get_digit(predicted_cat[0])         # get the digit name
    return predicted_digit, prediction_probs


LoD = [
    [0,0,0,8,14,0,0,0,0,0,5,16,11,0,0,0,0,1,15,14,1,6,0,0,0,7,16,5,3,16,8,0,0,8,16,8,14,16,2,0,0,0,6,14,16,11,0,0,0,0,0,6,16,4,0,0,0,0,0,10,15,0,0,0],
    [0,0,0,5,14,12,2,0,0,0,7,15,8,14,4,0,0,0,6,2,3,13,1,0,0,0,0,1,13,4,0,0,0,0,1,11,9,0,0,0,0,8,16,13,0,0,0,0,0,5,14,16,11,2,0,0,0,0,0,6,12,13,3,0],
    [0,0,0,3,16,3,0,0,0,0,0,12,16,2,0,0,0,0,8,16,16,4,0,0,0,7,16,15,16,12,11,0,0,8,16,16,16,13,3,0,0,0,0,7,14,1,0,0,0,0,0,6,16,0,0,0,0,0,0,4,14,0,0,0],
    [0,0,0,3,15,10,1,0,0,0,0,11,10,16,4,0,0,0,0,12,1,15,6,0,0,0,0,3,4,15,4,0,0,0,0,6,15,6,0,0,0,4,15,16,9,0,0,0,0,0,13,16,15,9,3,0,0,0,0,4,9,14,7,0],
    [0,0,0,3,16,3,0,0,0,0,0,10,16,11,0,0,0,0,4,16,16,8,0,0,0,2,14,12,16,5,0,0,0,10,16,14,16,16,11,0,0,5,12,13,16,8,3,0,0,0,0,2,15,3,0,0,0,0,0,4,12,0,0,0],
    [0,0,7,15,15,4,0,0,0,8,16,16,16,4,0,0,0,8,15,8,16,4,0,0,0,0,0,10,15,0,0,0,0,0,1,15,9,0,0,0,0,0,6,16,2,0,0,0,0,0,8,16,8,11,9,0,0,0,9,16,16,12,3,0]
]

# Run prediction on each test digit
print("\nTesting the digits from the assignment:")
for i, digit_features in enumerate(LoD):
    MODEL = nn_classifier
    SCALER = scaler
    name, probs = predictive_model(digit_features, MODEL, SCALER)
    print(f"Digit {i}: I predict {name}")
    
    # Visualize the digit
    digit_image = np.array(digit_features).reshape(8, 8)
    plt.figure(figsize=(2, 2))
    plt.imshow(digit_image, cmap='gray_r')
    plt.title(f"Digit {i}: Predicted as {name}")
    plt.show()



pixel_to_predict = 'pix42'
pixel_to_predict_idx = COL_INDEX[pixel_to_predict]


reg_y_all = A[:, pixel_to_predict_idx]



other_pixel_indices = [COL_INDEX[col] for col in pixel_cols if col != pixel_to_predict]
feature_indices = other_pixel_indices + label_indices



reg_X_all = A[:, feature_indices]

print(f"Regression y_all (pixel {pixel_to_predict} values) shape: {reg_y_all.shape}")
print(f"Regression X_all (other features) shape: {reg_X_all.shape}")


indices = np.random.permutation(len(reg_y_all))
reg_X_all = reg_X_all[indices]
reg_y_all = reg_y_all[indices]


reg_X_train, reg_X_test, reg_y_train, reg_y_test = train_test_split(reg_X_all, reg_y_all, test_size=0.2)

print(f"Regression training with {len(reg_y_train)} rows; testing with {len(reg_y_test)} rows\n")


reg_scaler = StandardScaler()
reg_scaler.fit(reg_X_train)

reg_X_train_scaled = reg_scaler.transform(reg_X_train)
reg_X_test_scaled = reg_scaler.transform(reg_X_test)


from sklearn.neural_network import MLPRegressor

nn_regressor = MLPRegressor(hidden_layer_sizes=(32, 16),
                     max_iter=500,
                     verbose=True,
                     shuffle=True,
                     random_state=42,
                     learning_rate_init=.05,
                     learning_rate='adaptive')


print("\n\n++++++++++  REGRESSION TRAINING:  begin  +++++++++++++++\n\n")
nn_regressor.fit(reg_X_train_scaled, reg_y_train)
print("\n++++++++++  REGRESSION TRAINING:   end  +++++++++++++++")
print(f"The (squared) prediction error (the loss) is {nn_regressor.loss_:<6.3f}")
print(f"And, its square root:         {nn_regressor.loss_ ** 0.5:<6.3f}")
print()


def ascii_table_for_regressor(Xsc, y, nn, scaler):
    """ a table including predictions using nn.predict """
    predictions = nn.predict(Xsc) # all predictions
    Xpr = scaler.inverse_transform(Xsc)  # Xpr is the "X to print": unscaled data!
    # measure error
    error = 0.0
    # printing
    print(f"{'sample #':>10s} ->  {'pred':^6s}  {'des.':^6s}   {'absdiff':^10s}")
    for i in range(min(20, len(y))):  # Show only first 20 to avoid large output
        pred = predictions[i]
        desired = y[i]
        result = abs(desired - pred)
        error += result
        print(f"{i:>10d} ->  {pred:<+6.2f}  {desired:<+6.2f}   {result:^10.2f}")

    error_full = 0.0
    for i in range(len(y)):
        pred = predictions[i]
        desired = y[i]
        error_full += abs(desired - pred)

    print("\n" + "+++++   +++++   +++++           ")
    print(f"average abs diff error:   {error_full/len(y):<6.3f}")
    print("+++++   +++++   +++++           ")
    return error_full/len(y)



print(f"\nTesting regression for pixel {pixel_to_predict}:")
avg_error = ascii_table_for_regressor(reg_X_test_scaled, reg_y_test, nn_regressor, reg_scaler)


def predict_pixel(pixel_name, A, pixel_indices, label_indices, test_size=0.2, hidden_layers=(32, 16), random_state=42, verbose=False):
    """
    Create and evaluate a neural network regressor to predict a specific pixel
    from all other pixels and digit classifications.
    
    Args:
        pixel_name: Name of the pixel to predict (e.g., 'pix42')
        A: The full data array including pixels and digit classifications
        pixel_indices: Dictionary mapping pixel names to column indices
        label_indices: Dictionary mapping label names to column indices
        test_size: Proportion of data to use for testing
        hidden_layers: Tuple defining the neural network architecture
        random_state: Random seed for reproducibility
        verbose: Whether to print detailed training progress
        
    Returns:
        avg_abs_error: Average absolute error for this pixel prediction
    """
    # Get the index of the pixel to predict
    pixel_idx = COL_INDEX[pixel_name]
    
    # Create X_all and y_all for regression
    pixel_y_all = A[:, pixel_idx]
    
    # Get indices of all other pixels and labels
    other_pixel_indices = [idx for idx in pixel_indices if idx != pixel_idx]
    all_feature_indices = other_pixel_indices + label_indices
    
    # Create features array (all other pixels + labels)
    pixel_X_all = A[:, all_feature_indices]
    
    if verbose:
        print(f"Regression y_all (pixel {pixel_idx} values) shape: {pixel_y_all.shape}")
        print(f"Regression X_all (other features) shape: {pixel_X_all.shape}")
    
    # Scramble the data
    indices = np.random.permutation(len(pixel_y_all))
    pixel_X_all = pixel_X_all[indices]
    pixel_y_all = pixel_y_all[indices]
    
    # Split into training and testing sets
    pixel_X_train, pixel_X_test, pixel_y_train, pixel_y_test = train_test_split(
        pixel_X_all, pixel_y_all, test_size=test_size, random_state=random_state)
    
    if verbose:
        print(f"Regression training with {len(pixel_y_train)} rows; testing with {len(pixel_y_test)} rows\n")
    
    # Scale the data
    pixel_scaler = StandardScaler()
    pixel_scaler.fit(pixel_X_train)
    
    pixel_X_train_scaled = pixel_scaler.transform(pixel_X_train)
    pixel_X_test_scaled = pixel_scaler.transform(pixel_X_test)
    
    # Train the regression model
    pixel_nn_regressor = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        max_iter=500,
        verbose=verbose,
        shuffle=True,
        random_state=random_state,
        learning_rate_init=.05,
        learning_rate='adaptive')
    
    if verbose:
        print(f"\n\n++++++++++  REGRESSION TRAINING FOR PIXEL {pixel_idx}:  begin  +++++++++++++++\n\n")
    
    pixel_nn_regressor.fit(pixel_X_train_scaled, pixel_y_train)
    
    if verbose:
        print(f"\n++++++++++  REGRESSION TRAINING FOR PIXEL {pixel_idx}:   end  +++++++++++++++")
        print(f"The (squared) prediction error (the loss) is {pixel_nn_regressor.loss_:<6.3f}")
        print(f"And, its square root:         {pixel_nn_regressor.loss_ ** 0.5:<6.3f}")
        print()
    
    # Calculate average absolute error
    predictions = pixel_nn_regressor.predict(pixel_X_test_scaled)
    abs_errors = [abs(predictions[i] - pixel_y_test[i]) for i in range(len(pixel_y_test))]
    avg_abs_error = sum(abs_errors) / len(abs_errors)
    
    if verbose:
        print(f"Average absolute error for pixel {pixel_idx}: {avg_abs_error:.4f}")
    
    return avg_abs_error



all_pixel_errors = []

print("\nEvaluating predictability of all 64 pixels...")
for pixel_num in range(64):
    pixel_name = f'pix{pixel_num}'
    error = predict_pixel(pixel_name, A, pixel_indices, label_indices, verbose=False)
    all_pixel_errors.append(error)
    if (pixel_num + 1) % 8 == 0:
        print(f"Processed pixels 0-{pixel_num}")


most_predictable = np.argmin(all_pixel_errors)
least_predictable = np.argmax(all_pixel_errors)
print(f"\nMost predictable pixel: {most_predictable} with error {all_pixel_errors[most_predictable]:.4f}")
print(f"Least predictable pixel: {least_predictable} with error {all_pixel_errors[least_predictable]:.4f}")


# Create a visualization of pixel predictability
pixel_errors_2d = np.array(all_pixel_errors).reshape(8, 8)

plt.figure(figsize=(10, 8))
plt.imshow(pixel_errors_2d, cmap='viridis_r')  # viridis_r so darker is less predictable
plt.colorbar(label='Average Absolute Error')
plt.title('Predictability of Each Pixel (Lower Value = More Predictable)')

# Add pixel indices as text in each cell
for i in range(8):
    for j in range(8):
        pixel_idx = i * 8 + j
        plt.text(j, i, str(pixel_idx), 
                 ha="center", va="center", 
                 color="white" if pixel_errors_2d[i, j] > np.mean(all_pixel_errors) else "black")

plt.savefig('pixel_predictability.png')
plt.show()

# Create a bar plot of all pixel errors
plt.figure(figsize=(12, 6))
plt.bar(range(64), all_pixel_errors)
plt.axhline(y=np.mean(all_pixel_errors), color='r', linestyle='-', label='Mean Error')
plt.xlabel('Pixel Index')
plt.ylabel('Average Absolute Error')
plt.title('Predictability of Each Pixel (Lower = More Predictable)')
plt.xticks(range(0, 64, 4))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('pixel_errors_bar.png')
plt.show()


