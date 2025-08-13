# 📦 Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Load dataset
df = pd.read_csv('Tharawat.csv')

# Show first rows
df.head()



# Show dataset structure and column types
df.info()


# Show basic stats like mean, min, max
df.describe()


# Select the features (input columns)
X = df[['Area_sqm', 'Num_Street_Faces', 'Street_Width']]

# Select the target column (output)
y = df['Price_SAR']

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Linear Regression model
model = LinearRegression()

# Fit the model with training data
model.fit(X_train, y_train)

# Make predictions using the test data
y_pred = model.predict(X_test)

# Show first 5 predictions vs actual prices
print("Predicted Prices:", y_pred[:5])
print("Actual Prices:   ", y_test.values[:5])



from sklearn.metrics import r2_score

# Calculate the R-squared score
r2 = r2_score(y_test, y_pred)

# Print the result
print("R-squared Score:", r2)


# Scatter plot: real vs predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect prediction line
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()



# Remove 'Area_sqm' and test the model
X = df[['Num_Street_Faces', 'Street_Width']]
y = df['Price_SAR']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print("R-squared without Area_sqm:", r2)



# Scatter plot for model without 'Area_sqm'
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect prediction line
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted (Without Area_sqm)")
plt.grid(True)
plt.show()



# Remove 'Num_Street_Faces' and test the model
X = df[['Area_sqm', 'Street_Width']]
y = df['Price_SAR']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print("R-squared without Num_Street_Faces:", r2)



# Scatter plot for model without 'Num_Street_Faces'
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect prediction line
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted (Without Num_Street_Faces)")
plt.grid(True)
plt.show()



# Remove 'Street_Width' and test the model
X = df[['Area_sqm', 'Num_Street_Faces']]
y = df['Price_SAR']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print("R-squared without Street_Width:", r2)



# Scatter plot for model without 'Street_Width'
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect prediction line
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted (Without Street_Width)")
plt.grid(True)
plt.show()



# Step 6 – Modeling with KNN Regressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Use the same features
X = df[['Area_sqm', 'Num_Street_Faces', 'Street_Width']]
y = df['Price_SAR']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the KNN model
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_knn = knn_model.predict(X_test)
r2_knn = r2_score(y_test, y_pred_knn)
print("R-squared Score (KNN):", r2_knn)

# Visualize the predictions
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_knn, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("KNN Regressor: Actual vs Predicted Prices")
plt.grid(True)
plt.show()



# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv('Tharawat.csv')

# Define features and target
features = ['Area_sqm', 'Num_Street_Faces', 'Street_Width']
target = 'Price_SAR'

X = df[features]
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Train KNN Regressor model
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# Compare visualizations
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Plot Linear Regression
axs[0].scatter(y_test, y_pred_lr, color='blue')
axs[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axs[0].set_title("Linear Regression")
axs[0].set_xlabel("Actual Prices")
axs[0].set_ylabel("Predicted Prices")
axs[0].grid(True)

# Plot KNN Regressor
axs[1].scatter(y_test, y_pred_knn, color='purple')
axs[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axs[1].set_title("KNN Regressor")
axs[1].set_xlabel("Actual Prices")
axs[1].set_ylabel("Predicted Prices")
axs[1].grid(True)

plt.tight_layout()
plt.show()



