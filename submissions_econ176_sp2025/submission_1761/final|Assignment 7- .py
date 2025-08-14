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


# Generating Histogram of Driver Age

plt.hist(df_all['driver_age'])
plt.xlabel('Driver Age')
plt.ylabel('Frequency')
plt.title('Histogram of Driver Ages')
plt.show()


# Histogram of Car Ages
plt.hist(df_all['car_age'])
plt.xlabel('Car Age')
plt.ylabel('Frequency')
plt.title('Histogram of Car Ages')
plt.show()


# a histogram of the claim numbers
plt.hist(df_all['claim_nb'])
plt.xlabel('Claim Number')
plt.ylabel('Frequency')
plt.title('Histogram of Claim Numbers')
plt.show()


# create a scatter plot of how the car's age varies with the driver's age
plt.scatter(df_all['driver_age'], df_all['car_age'])
plt.xlabel('Driver Age')
plt.ylabel('Car Age')
plt.title('Scatter Plot of Driver vs. Car Age')
plt.show()


# create a scatter plot of how the claim_nb varies with the driver's age
plt.scatter(df_all['driver_age'], df_all['claim_nb'])
plt.xlabel('Driver Age')
plt.ylabel('Claim Number')
plt.title('Scatter Plot of Driver vs. Claim Number')
plt.show()


# plot of my choice: scatter plot of car weight and car power
plt.scatter(df_all['car_weight'], df_all['car_power'])
plt.xlabel('Car Weight')
plt.ylabel('Car Power')
plt.title('Scatter Plot of Car Weight vs. Car Power')
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
from glum import Gener[REDACTED]zed[REDACTED]nearRegressor

glm_model = Gener[REDACTED]zed[REDACTED]nearRegressor(family="poisson", alpha=1e-6)
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


# Computing the per-unit incident rate multiplier for car_power

incident_rate_multiplier = exp(-0.004117)
print("Incident Rate Multiplier for car_power:", incident_rate_multiplier)

# On average, cars have 124.981766 Horse Power
incident_horsepower = incident_rate_multiplier ** 125
print("Incident_horsepower:", incident_horsepower)

# for full span of dataset
full_span = 341 - 50
full_span_rate = incident_rate_multiplier ** full_span
print("Incident Rate Multiplier for Full Span:", full_span_rate)

# Interpretation: Those with 341 horsepower cars will make 0.3 times the claims of those with 50 horsepower cars.
# This does not suport my intution. I expected those with higher horsepower cars to be riskier drivers and file more claims.



# Computing the per-unit incident rate multiplier for car_age
incident_rate_multiplier = exp(-0.022168)
print("Incident Rate Multiplier for car_age:", incident_rate_multiplier)

# On average, cars have 4.15 age
incident_age = incident_rate_multiplier ** 4.15
print("Car age multiplier:", incident_age)

# for full span of dataset
full_span = 23 - 0
full_span_rate = incident_rate_multiplier ** full_span
print("Incident Rate Multiplier for Full Span:", full_span_rate)

# Interpretation: Those with car ages of 23 years filed 0.97 more claims than owners of cares with ages of 0. This goes against
# my intution, as I'd expect older cars to have more issues and require more insurance claims than relatively newer cars.


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
    v[REDACTED]dation_split=0.1,
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
    v[REDACTED]dation_split=0.1,
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

# Add hidden layers (step-down architecture)
h1 = layers.Dense(32, activation="relu")(inputs)  # First hidden layer with 32 neurons and ReLU activation
h2 = layers.Dense(16, activation="relu")(h1)  # Second hidden layer with 16 neurons and ReLU activation
h3 = layers.Dense(8, activation="relu")(h2)   # Third hidden layer with 8 neurons and ReLU activation

outputs = layers.Dense(1, activation="exponential")(h3)
nn_model_deep = keras.Model(inputs=inputs, outputs=outputs)

nn_model_deep.summary()

# Calculate gradients
nn_model_deep.compile(optimizer=Adam(learning_rate=1e-4), loss="Poisson")


tf.random.set_seed(4349)

history_2layer = nn_model_deep.fit(
    x=X_train,
    y=train[y],
    epochs=200,
    batch_size=10_000,
    v[REDACTED]dation_split=0.1,
    callbacks=cb,
    verbose=1,         # consider running both values 0 (no printing) and 1 (printing)
)


# run SHAP
def nn_predict_2layer(X):
    """Prediction function of the neural network (log scale)."""
    df = pd.DataFrame(X, columns=x)
    df_scaled = nn_preprocessor.transform(df)
    pred = nn_model_deep.predict(df_scaled, verbose=0, batch_size=10_000).flatten()
    return np.log(pred)

nn_explainer = shap.KernelExplainer(nn_predict_2layer, data=X_bg)
shap_nn = nn_explainer.shap_values(X_explain, nsamples=30)

all_dep_plots(x, shap_nn, X_explain)   # create all feature-dependency plots


