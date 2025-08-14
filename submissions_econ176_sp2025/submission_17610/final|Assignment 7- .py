# some of the libraries have warnings we want to ignore...
import warnings
warnings.filterwarnings("ignore")

# AI/ML libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml




# Fetch the insurance-claim dataset from the OpenML repository  (an API call! :)
all = fetch_openml(data_id=45106, as_frame=True)


# Extract the pieces of the API result from the OpenML API call
df_all = all.data
df_all["claim_nb"] = all.target

print("Shape of full data:", df_all.shape)
df_all.head(15)    # print the first five rows of data


df_all.describe()


#
# It's easiest to make additional code cells...
#


# Histogram of drivers' ages
plt.figure(figsize=(10, 6))
plt.hist(df_all['driver_age'], bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribution of Driver Ages', fontsize=14)
plt.xlabel('Driver Age (years)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# Histogram of cars' ages
plt.figure(figsize=(10, 6))
plt.hist(df_all['car_age'], bins=25, alpha=0.7, color='green', edgecolor='black')
plt.title('Distribution of Car Ages', fontsize=14)
plt.xlabel('Car Age (years)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(alpha=0.3)
plt.show()


# Histogram of claim numbers
plt.figure(figsize=(10, 6))
plt.hist(df_all['claim_nb'], bins=np.arange(-0.5, 5.5, 1), alpha=0.7, color='red', edgecolor='black')
plt.title('Distribution of Insurance Claims', fontsize=14)
plt.xlabel('Number of Claims', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(range(0, 5))
plt.grid(alpha=0.3)
plt.show()

# Scatter plot of car age vs driver age
plt.figure(figsize=(10, 6))
# Take a random sample to avoid overplotting
sample_size = 50000
sample_df = df_all.sample(sample_size, random_state=42)
plt.scatter(sample_df['driver_age'], sample_df['car_age'], alpha=0.3, s=10)
plt.title('Car Age vs Driver Age', fontsize=14)
plt.xlabel('Driver Age (years)', fontsize=12)
plt.ylabel('Car Age (years)', fontsize=12)
plt.grid(alpha=0.3)
plt.show()


# Scatter plot of claim_nb vs driver age
plt.figure(figsize=(10, 6))
# Use jittering to better visualize discrete claim values
claims_df = df_all.sample(sample_size, random_state=24)
plt.scatter(claims_df['driver_age'], claims_df['claim_nb'] + np.random.normal(0, 0.05, len(claims_df)), 
            alpha=0.3, s=10, color='purple')
plt.title('Number of Claims vs Driver Age', fontsize=14)
plt.xlabel('Driver Age (years)', fontsize=12)
plt.ylabel('Number of Claims', fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# Additional scatter plot: car_power vs claim_nb with town as color
plt.figure(figsize=(10, 6))
power_claims_df = df_all.sample(sample_size, random_state=35)
scatter = plt.scatter(power_claims_df['car_power'], 
                     power_claims_df['claim_nb'] + np.random.normal(0, 0.05, len(power_claims_df)),
                     c=power_claims_df['town'], cmap='viridis', 
                     alpha=0.5, s=15)
plt.colorbar(scatter, label='Town (1=urban, 0=rural)')
plt.title('Number of Claims vs Car Power (colored by Town)', fontsize=14)
plt.xlabel('Car Power (hp)', fontsize=12)
plt.ylabel('Number of Claims', fontsize=12)
plt.grid(alpha=0.3)
plt.show()


# Let's make sure everything runs first... using only 100,000 rows (instead of 1,000,000)

df_all = df_all.sample(frac=1.0, random_state=42)  # This shuffles the 1,000,000-row dataset

NUMROWS = 100000
df = df_all.iloc[0:NUMROWS,:].copy()    # This uses only the first 100,000 rows (or NUMROWS rows) for speed...


# We split into 90% training data and 10% testing data

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.1, random_state=30)
print("Shape of training data:", train.shape)
print("Shape of test data:", test.shape)


# Define target name (y) and feature names (x)
y, x = df.columns[-1], list(df.columns[:-1])

print("The response name:", y)
print("The feature names:", x)





#
# there's not much code to build our model!
#
from glum import GeneralizedLinearRegressor

glm_model = GeneralizedLinearRegressor(family="poisson", alpha=1e-6)
glm_model.fit(X=train[x], y=train[y])

print("Coefficients")
pd.Series(np.append(glm_model.intercept_, glm_model.coef_), index=["Intercept"] + x)


from math import exp
exp(0.36)


exp(0.360105)    # for each unit of "townness,"  we multiply the expected claims by 1.433:


exp(0.360105) ** 2   # so, we multiply twice for _two_ units of "townness":


exp(-0.003272)    # for each year of driver_age, you multiply the expected claims by 0.9967:


0.9967 ** 15


exp(-0.003272)**(88-18)


1/0.7952


from math import exp

print("Car Power Analysis:")
print(f"Per unit multiplier: {exp(0.004117)}")
print(f"50hp increase: {exp(0.004117 * 50)}")
print(f"Full range (50hp to 341hp): {exp(0.004117 * (341-50))}")

print("Car Weight Analysis:")
print(f"Per unit (kg) multiplier: {exp(-0.000068)}")
print(f"500kg increase: {exp(-0.000068 * 500)}")
print(f"Full range (950kg to 3120kg): {exp(-0.000068 * (3120-950))}")


import shap


import shap

# First, extract background data. This is the same for all models to interpret:
X_bg = train[x].sample(200, random_state=8366)    # grab 200 samples from our training data - 200 is a lot...

# Exploring the space...   This will take a while...
glm_explainer = shap.KernelExplainer(lambda x: np.log(glm_model.predict(x)), data=X_bg)     # Don't worry about the warnings!

# Then, we can choose any data to explain. We'll choose 1,000 rows randomly called X_explain
X_explain = train[x].sample(n=1000, random_state=937)
shap_glm = glm_explainer.shap_values(X_explain, nsamples=30) # there are 30 non-trivial subsets of 6 features


X_explain[0:1]


shap_glm[0:1]


#
# A function to create all of the dependence plots among our features
#
def all_dep_plots(x, shap_values, X):
    """ Dependence plots for all features x. """
    fig, _ = plt.subplots(nrows=2, ncols=3, figsize=(10, 6), sharey=True)

    for i, ax in enumerate(fig.axes):
        xvar = x[i]
        shap.dependence_plot(
            xvar,
            shap_values,
            features=X,
            x_jitter=0.2 * (xvar in ("town", "year")),
            ymin=-0.5,
            ymax=1,
            ax=ax,
            show=False,
        )
        ax.set_title(xvar, fontdict={"size": 16})
        ax.set_ylabel("SHAP values" if i % 3 == 0 else "")
        ax.grid()
    plt.tight_layout()

print("The all_dep_plots function has been defined.")
print("Run the next cell to plot the features and their impacts")


all_dep_plots(x, shap_glm, X_explain)


# Scale features to [-1, 1]
from sklearn.preprocessing import MinMaxScaler

nn_preprocessor = MinMaxScaler(feature_range=(-1, 1))
X_train = nn_preprocessor.fit_transform(train[x])

print("Output after the data has been scaled:")
pd.DataFrame(X_train[0:5], columns=x)


#
# Neural nets are worthy of their own course, for sure...
#
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
cb = [EarlyStopping(patience=20), ReduceLROnPlateau(patience=5)]  # "callbacks" for training (see below)

# Architecture
inputs = keras.Input(shape=(len(x),))
# additional layers can be added here!
outputs = layers.Dense(1, activation="exponential")(inputs)
nn_model_shallow = keras.Model(inputs=inputs, outputs=outputs)

nn_model_shallow.summary()

# Calculate gradients
nn_model_shallow.compile(optimizer=Adam(learning_rate=1e-4), loss="Poisson")


#
# train the network!
#

tf.random.set_seed(4349)

history_shallow = nn_model_shallow.fit(
    x=X_train,
    y=train[y],
    epochs=200,
    batch_size=10_000,
    validation_split=0.1,
    callbacks=cb,
    verbose=1,         # consider running both values 0 (no printing) and 1 (printing)
)


# prompt: Could you show a heatmap of the values of the neuron's weights in the above model, named nn_model?

import seaborn as sns
import matplotlib.pyplot as plt

# Here, nn_model_shallow is the Keras (NNet) model
weights = nn_model_shallow.layers[1].get_weights()[0]  # Get weights from this layer (omitting the intercepts, which are in [1])
weights = np.transpose(weights)

# Create a heatmap of the weights
plt.figure(figsize=(10, 8))
sns.heatmap(weights, annot=False, cmap='viridis')
plt.title("Neuron Weights Heatmap of the Input Layer")
plt.xlabel("Input Features (6)")
plt.ylabel("Neurons from Inputs to Output")
plt.show()


def nn_predict_shallow(X):
    """Prediction function of the neural network (log scale)."""
    df = pd.DataFrame(X, columns=x)
    df_scaled = nn_preprocessor.transform(df)
    pred = nn_model_shallow.predict(df_scaled, verbose=0, batch_size=10_000).flatten()
    return np.log(pred)


nn_explainer = shap.KernelExplainer(nn_predict_shallow, data=X_bg)
shap_nn = nn_explainer.shap_values(X_explain, nsamples=30)

all_dep_plots(x, shap_nn, X_explain)


#
# Let's add a second layer to our network:

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
cb = [EarlyStopping(patience=20), ReduceLROnPlateau(patience=5)]  # "callbacks" for training (see below)

# Architecture: Adding Layers
inputs = keras.Input(shape=(len(x),))            # Here is the input layer

z = layers.Dense(7, activation="tanh")(inputs)   # Here is the new layer: 7 neurons, inputs are the input, and z is the output. Use "tanh"

# For more layers, you can continue using z.     # Here is a commented-out example:
# z = layers.Dense(3, activation="tanh"))(z)     # This is another layer: 3 neurons, z is the input, and z is the output

outputs = layers.Dense(1, activation="exponential")(z)   # Here, we convert the previous layer's results (z) into the overall output
nn_model_2layer = keras.Model(inputs=inputs, outputs=outputs)   # The final layer often uses a different activation function

nn_model_2layer.summary()

nn_model_2layer.compile(optimizer=Adam(learning_rate=1e-4), loss="Poisson")


tf.random.set_seed(4349)

history_2layer = nn_model_2layer.fit(
    x=X_train,
    y=train[y],
    epochs=200,
    batch_size=10_000,
    validation_split=0.1,
    callbacks=cb,
    verbose=1,         # consider running both values 0 (no printing) and 1 (printing)
)


import seaborn as sns
import matplotlib.pyplot as plt

weights = nn_model_2layer.layers[1].get_weights()[0]  # Get weights from this layer (omit the intercept, which is [1])
weights = np.transpose(weights)

# Create a heatmap of the weights
plt.figure(figsize=(10, 8))
sns.heatmap(weights, annot=False, cmap='viridis')
plt.title("Neuron Weights Heatmap of the Input Layer")
plt.xlabel(f"Input Features (6)")
plt.ylabel("Neurons from Inputs to Layer1 (7)")
plt.show()



# Next layer
weights = nn_model_2layer.layers[2].get_weights()[0]  # Get weights from this layer (omit the intercept, which is [1])
weights = np.transpose(weights)

# Create a heatmap of the weights
plt.figure(figsize=(10, 8))
sns.heatmap(weights, annot=False, cmap='viridis')
plt.title("Neuron Weights Heatmap of Next Layer")
plt.xlabel("Neurons from Layer 1 (of 7)")
plt.ylabel("Neurons from Layer 1 to Layer 2 (of 1)")
plt.show()



def nn_predict_2layer(X):
    """Prediction function of the neural network (log scale)."""
    df = pd.DataFrame(X, columns=x)
    df_scaled = nn_preprocessor.transform(df)
    pred = nn_model_2layer.predict(df_scaled, verbose=0, batch_size=10_000).flatten()
    return np.log(pred)

nn_explainer = shap.KernelExplainer(nn_predict_2layer, data=X_bg)
shap_nn = nn_explainer.shap_values(X_explain, nsamples=30)

all_dep_plots(x, shap_nn, X_explain)   # create all feature-dependency plots


#
# The age is a much more complicated - non-linear - feature
def age_effect(age):
    x = (age - 66) / 60
    return 0.05 + x**8 + 0.4 * x**3 + 0.3 * x**2 + 0.06 * x

#
# here is the true model for this dataset:
def true_model(X):
    """Returns pd.Series of true expected frequencies."""
    df = pd.DataFrame(X, columns=x)  # Needed because SHAP turns df to np.array
    log_lambda = (
        0.15 * df.town
        + np.log(age_effect(df.driver_age))
        + (0.3 + 0.15 * df.town) * df.car_power / 100
        - 0.02 * df.car_age
    )
    return np.exp(log_lambda)


import shap

true_model_explainer = shap.KernelExplainer(lambda x: np.log(true_model(x)), data=X_bg)
shap_true = true_model_explainer.shap_values(X_explain, nsamples=30)

all_dep_plots(x, shap_true, X_explain)


#

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
cb = [EarlyStopping(patience=20), ReduceLROnPlateau(patience=5)]  # "callbacks" for training (see below)

inputs = keras.Input(shape=(len(x),))

#  more neurons! 
z = layers.Dense(24, activation="relu")(inputs) 

z = layers.Dense(12, activation="relu")(z)

z = layers.Dense(6, activation="relu")(z)  

# Output layer - keep exponential activation for Poisson distribution
outputs = layers.Dense(1, activation="exponential")(z)
nn_model_stepdown = keras.Model(inputs=inputs, outputs=outputs)

nn_model_stepdown.summary()

nn_model_stepdown.compile(optimizer=Adam(learning_rate=2e-4), loss="Poisson")


history_stepdown = nn_model_stepdown.fit(
    x=X_train,
    y=train[y],
    epochs=200,
    batch_size=10_000,
    validation_split=0.1,
    callbacks=cb,
    verbose=1
)


def nn_predict_stepdown(X):
    """Prediction function of the step-down neural network (log scale)."""
    df = pd.DataFrame(X, columns=x)
    df_scaled = nn_preprocessor.transform(df)
    pred = nn_model_stepdown.predict(df_scaled, verbose=0, batch_size=10_000).flatten()
    return np.log(pred)

# Create the explainer
nn_stepdown_explainer = shap.KernelExplainer(nn_predict_stepdown, data=X_bg)

# Calculate SHAP values (this may take a while for deeper networks)
shap_nn_stepdown = nn_stepdown_explainer.shap_values(X_explain, nsamples=30)

# Plot dependency plots using the existing function
all_dep_plots(x, shap_nn_stepdown, X_explain)


# Training history plot

# After training the model
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_stepdown.history['loss'], label='Training Loss')
plt.plot(history_stepdown.history['val_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

if 'lr' in history_stepdown.history:
    plt.subplot(1, 2, 2)
    plt.plot(history_stepdown.history['lr'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    
plt.tight_layout()
plt.show()


# Prediction vs actual

# Test set predictions
X_test = nn_preprocessor.transform(test[x])
y_pred = nn_model_stepdown.predict(X_test, verbose=0).flatten()
y_true = test[y].values

# Simple metrics
mean_abs_error = np.mean(np.abs(y_pred - y_true))
mean_squared_error = np.mean((y_pred - y_true)**2)
print(f"Mean Absolute Error: {mean_abs_error:.4f}")
print(f"Mean Squared Error: {mean_squared_error:.4f}")

# Plot prediction distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(y_true, bins=10, alpha=0.5, label='Actual')
plt.hist(y_pred, bins=10, alpha=0.5, label='Predicted')
plt.legend()
plt.title('Distribution Comparison')

plt.subplot(1, 2, 2)
plt.scatter(y_true, y_pred, alpha=0.3)
plt.plot([0, max(y_true)], [0, max(y_true)], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.tight_layout()
plt.show()


# attempt after many iterations

#

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
cb = [EarlyStopping(patience=20), ReduceLROnPlateau(patience=5)]

inputs = keras.Input(shape=(len(x),))


z = layers.Dense(24)(inputs)  
z = layers.LeakyReLU(alpha=0.1)(z)  
z = layers.Dropout(0.25)(z) 

# Second layer
z = layers.Dense(12)(z)
z = layers.LeakyReLU(alpha=0.1)(z)

# Third layer
z = layers.Dense(6)(z)        
z = layers.LeakyReLU(alpha=0.1)(z)

# Output layer
outputs = layers.Dense(1, activation="exponential")(z)
nn_model_improved = keras.Model(inputs=inputs, outputs=outputs)

nn_model_improved.summary()

# Keep same learning rate
nn_model_improved.compile(optimizer=Adam(learning_rate=2e-4), loss="Poisson")


history_improved = nn_model_improved.fit(
    x=X_train,
    y=train[y],
    epochs=200,
    batch_size=5000,
    validation_split=0.1,
    callbacks=cb,
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.plot(history_improved.history['loss'], label='Training Loss')
plt.plot(history_improved.history['val_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluation on test set
X_test = nn_preprocessor.transform(test[x])
y_pred = nn_model_improved.predict(X_test, verbose=0).flatten()
y_true = test[y].values

# Metrics
mean_abs_error = np.mean(np.abs(y_pred - y_true))
mean_squared_error = np.mean((y_pred - y_true)**2)
print(f"Mean Absolute Error: {mean_abs_error:.4f}")
print(f"Mean Squared Error: {mean_squared_error:.4f}")

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(y_true, bins=10, alpha=0.5, label='Actual')
plt.hist(y_pred, bins=10, alpha=0.5, label='Predicted')
plt.legend()
plt.title('Distribution Comparison')

plt.subplot(1, 2, 2)
plt.scatter(y_true, y_pred, alpha=0.3)
plt.plot([0, max(y_true)], [0, max(y_true)], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.tight_layout()
plt.show()

# SHAP Analysis
def nn_predict_improved(X):
    """Prediction function of the improved neural network (log scale)."""
    df = pd.DataFrame(X, columns=x)
    df_scaled = nn_preprocessor.transform(df)
    pred = nn_model_improved.predict(df_scaled, verbose=0, batch_size=10_000).flatten()
    return np.log(pred)

# Create explainer
nn_improved_explainer = shap.KernelExplainer(nn_predict_improved, data=X_bg)
shap_nn_improved = nn_improved_explainer.shap_values(X_explain, nsamples=30)

# Plot dependency plots
all_dep_plots(x, shap_nn_improved, X_explain)

# Feature importance summary
feature_importance = np.abs(shap_nn_improved).mean(0)
features_df = pd.DataFrame(list(zip(x, feature_importance)), columns=['Feature', 'Importance'])
features_df = features_df.sort_values('Importance', ascending=False)
print("Feature Importance:")
print(features_df)


