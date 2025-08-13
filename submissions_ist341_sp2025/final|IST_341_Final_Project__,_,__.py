import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import accuracy_score

# Load dataset
file_path = "AB_NYC_2019.csv"
data = pd.read_csv(file_path)


# Display the dataset in a nice tabular format (without print)
data.head(20)


# Drop uninformative columns
data = data.drop(columns=['id', 'host_id', 'name', 'host_name', 'longitude','latitude','last_review','neighbourhood_group','calculated_host_listings_count','minimum_nights'])

# Drop rows with missing values (or impute if needed)
data = data.dropna()

# Encode categorical features
label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define features (X) and target (y)
X = data.drop(columns=['price'])
y = data['price']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





# Distribution of Prices

import seaborn as sns
import matplotlib.pyplot as plt

# Limit to reasonable prices (e.g., under $1000)
filtered_price_data = data[data['price'] <= 1000]

plt.figure(figsize=(12, 6))
sns.kdeplot(data=filtered_price_data, x="price", fill=True, color="green", alpha=0.7)

plt.title("Distribution of Price for NYC Airbnb Listings (Under $1000)")
plt.xlabel("Price (USD)")
plt.ylabel("Density")
plt.show()


# Visualizing Feature Relationships with a Correlation Heatmap (Airbnb Dataset)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Create a correlation matrix from the Airbnb dataset
corr_matrix = data.corr()

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", linewidths=.5, ax=ax)
plt.title("Correlation Heatmap of NYC Airbnb Listing Features")
plt.show()



# Decision Tree
dtree_model = tree.DecisionTreeClassifier()
dtree_model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, dtree_model.predict(X_test)) * 100
print('accuracy ', accuracy )


# Finding the Best Number of Neighbors for KNN

best_k = None
best_accuracy = 0.0

for k in range(1, 8):
    knn_cv_model = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn_cv_model, X_train, y_train, cv=5)
    average_cv_accuracy = cv_scores.mean()
    print(f"k: {k:2d}  cv accuracy: {average_cv_accuracy:7.4f}")
    if average_cv_accuracy > best_accuracy:
        best_accuracy = average_cv_accuracy
        best_k = k

print(f"\nBest k = {best_k} yields the highest average CV accuracy.")

# Train final KNN model and evaluate
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)
print("Test set accuracy:", knn_model.score(X_test, y_test) * 100)


# KNeighbors model
knn_model = KNeighborsClassifier(n_neighbors=best_k)   # here, we use the best_k!
knn_model.fit(X_train, y_train)
print("Accuracy on test set:", knn_model.score(X_test, y_test)* 100 )



# Finding the Best Random Forest Depth and Number of Trees

best_d = 1
best_ntrees = 50
best_accuracy = 0

for d in range(9, 10):
    for ntrees in range(50, 300, 100):
        rforest_model = ensemble.RandomForestClassifier(
            max_depth=d,
            n_estimators=ntrees,
            max_samples=0.5
        )
        cv_scores = cross_val_score(rforest_model, X_train, y_train, cv=5)
        average_cv_accuracy = cv_scores.mean()
        print(f"depth: {d:2d} ntrees: {ntrees:3d} cv accuracy: {average_cv_accuracy:7.4f}")

        if average_cv_accuracy > best_accuracy:
            best_d = d
            best_ntrees = ntrees

best_depth = best_d
best_num_trees = best_ntrees

print(f"\nBest depth: {best_depth}, Best number of trees: {best_num_trees}")



# Random Forest:
rforest_model= ensemble.RandomForestClassifier(max_depth=best_depth,
                                                      n_estimators=best_num_trees,
                                                      max_samples=0.5)
rforest_model.fit(X_train, y_train)
print(f"Built an RF classifier with depth={best_depth} and ntrees={best_num_trees}")
print("Accuracy on test set:", rforest_model.score(X_test, y_test) * 100)


print("Accuracy on test set:", rforest_model.score(X_test, y_test)*100)


# Neural network
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

scaler = StandardScaler(with_mean=False)
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_scaled = y_train
y_test_scaled = y_test

nn_classifier = MLPClassifier(
    hidden_layer_sizes=(5, 10),
    max_iter=500,
    activation="tanh",
    solver='sgd',
    verbose=True,
    shuffle=True,
    random_state=None,
    learning_rate_init=0.1,
    learning_rate='adaptive'
)

print("Training started...")
nn_classifier.fit(X_train_scaled, y_train_scaled)
print("Training completed.")
print("Final training loss:", nn_classifier.loss_)
print("Test accuracy:", nn_classifier.score(X_test_scaled, y_test_scaled) * 100)


# Load dataset
file_path = "Student_performance_data.csv"
data = pd.read_csv(file_path)


# Display the dataset in a nice tabular format (without print)
data.head(20)


# Drop uninformative columns
data = data.drop(columns=['StudentID'])



# Drop rows with missing values (or impute if needed)
data = data.dropna()

# Encode categorical features
label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define features (X) and target (y)
X = data.drop(columns=['GradeClass'])
y = data['GradeClass']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





# Distribution of GPA for All Students

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = 12, 6

# Plot GPA density for all students
sns.kdeplot(data=data, x="GPA", fill=True, color="skyblue", alpha=0.7)

# Format plot
plt.title("Distribution of GPA for All Students")
plt.xlabel("GPA")
plt.ylabel("Density")
plt.show()


# Visualizing Feature Relationships with a Correlation Heatmap


# Making sure a provided example works, adapted to student data...

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Create a correlation matrix from the student dataset
corr_matrix = data.corr()

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", linewidths=.5, ax=ax)
plt.title("Correlation Heatmap of Student Performance Features")
plt.show()


# This heatmap visualizes the correlation between numerical features in the student performance dataset.
# Stronger correlations are indicated by higher absolute values and more intense colors.
# GPA shows a positive correlation with study time and parental support, and a negative correlation with absences.
# This visualization helps identify which factors are most associated with student academic performance.



# Descion Tree model:

# Train a Decision Tree model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
accuracy = 100 - (mae / y_test.mean()) * 100
print("Approximate accuracy (%):", accuracy)


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

best_k = None
best_score = -np.inf  # For maximizing R^2

for k in range(1, 8):
    knn_cv_model = KNeighborsRegressor(n_neighbors=k)
    cv_scores = cross_val_score(knn_cv_model, X_train, y_train, cv=5, scoring='r2')
    average_cv_score = cv_scores.mean()
    print(f"k: {k:2d}  cv R² score: {average_cv_score:7.4f}")
    if average_cv_score > best_score:
        best_score = average_cv_score
        best_k = k

print(f"\nBest k = {best_k} yields the highest average CV R² score.")

# Train final KNN regressor and evaluate
knn_model = KNeighborsRegressor(n_neighbors=best_k)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)

print("Test set R² score:", r2_score(y_test, y_pred))
print("Test set RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))



# Finding the Best Number of Neighbors for KNN

best_k = None
best_accuracy = 0.0

for k in range(1, 8):
    knn_cv_model = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn_cv_model, X_train, y_train, cv=5)
    average_cv_accuracy = cv_scores.mean()
    print(f"k: {k:2d}  cv accuracy: {average_cv_accuracy:7.4f}")
    if average_cv_accuracy > best_accuracy:
        best_accuracy = average_cv_accuracy
        best_k = k

print(f"\nBest k = {best_k} yields the highest average CV accuracy.")

# Train final KNN model and evaluate
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)
print("Test set accuracy:", knn_model.score(X_test, y_test) * 100)



# Finding the Best Random Forest Depth and Number of Trees

best_d = 1
best_ntrees = 50
best_accuracy = 0

for d in range(9, 10):
    for ntrees in range(50, 300, 100):
        rforest_model = ensemble.RandomForestClassifier(
            max_depth=d,
            n_estimators=ntrees,
            max_samples=0.5
        )
        cv_scores = cross_val_score(rforest_model, X_train, y_train, cv=5)
        average_cv_accuracy = cv_scores.mean()
        print(f"depth: {d:2d} ntrees: {ntrees:3d} cv accuracy: {average_cv_accuracy:7.4f}")

        if average_cv_accuracy > best_accuracy:
            best_d = d
            best_ntrees = ntrees

best_depth = best_d
best_num_trees = best_ntrees

print()
print(f"Best depth: {best_depth}, Best number of trees: {best_num_trees}")



# Random Forest:
rforest_model= ensemble.RandomForestClassifier(max_depth=best_depth,
                                                      n_estimators=best_num_trees,
                                                      max_samples=0.5)
rforest_model.fit(X_train, y_train)
print(f"Built an RF classifier with depth={best_depth} and ntrees={best_num_trees}")
print("Accuracy on test set:", rforest_model.score(X_test, y_test) * 100)


from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

scaler = StandardScaler(with_mean=False)
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_scaled = y_train
y_test_scaled = y_test

nn_classifier = MLPClassifier(
    hidden_layer_sizes=(5, 10),
    max_iter=500,
    activation="tanh",
    solver='sgd',
    verbose=True,
    shuffle=True,
    random_state=None,
    learning_rate_init=0.1,
    learning_rate='adaptive'
)

print("Training started...")
nn_classifier.fit(X_train_scaled, y_train_scaled)
print("Training completed.")
print("Final training loss:", nn_classifier.loss_)
print("Test accuracy:", nn_classifier.score(X_test_scaled, y_test_scaled) * 100)


