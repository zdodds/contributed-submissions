import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

#



df = pd.read_csv("births_cleaned.csv") 


X = df[['day', 'month']]
y = (df['births'] > df['births'].median()).astype(int) 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





dt_model = DecisionTreeClassifier(max_depth=2, random_state=42) 
dt_model.fit(X_train, y_train)


plt.figure(figsize=(10, 6))
plot_tree(dt_model, feature_names=X.columns, class_names=['Below Median', 'Above Median'], filled=True)
plt.show()

best_depth = None
best_score = 0

for depth in range(1, 6):
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(dt, X_train, y_train, cv=5)
    mean_score = scores.mean()
    
    if mean_score > best_score:
        best_score = mean_score
        best_depth = depth

print(f"Best Decision Tree Depth: {best_depth}")

#



best_depth_rf = None
best_num_trees = None
best_rf_score = 0

for depth in range(1, 6):
    for num_trees in [50, 150, 250]:
        rf = RandomForestClassifier(n_estimators=num_trees, max_depth=depth, random_state=42)
        scores = cross_val_score(rf, X_train, y_train, cv=5)
        mean_score = scores.mean()

        if mean_score > best_rf_score:
            best_rf_score = mean_score
            best_depth_rf = depth
            best_num_trees = num_trees

print(f"Best RF Depth: {best_depth_rf}, Best Num Trees: {best_num_trees}")



rf_model = RandomForestClassifier(n_estimators=best_num_trees, max_depth=best_depth_rf, random_state=42)
rf_model.fit(X_train, y_train)



#
# #Our model indicates that month/day both have a huge impact on birth trends (seasonal patterns or days with high birth rate). However, real world factors like cultural norms or big events happening in the world that play big roles are not considered. We should not rely too much on the model "outside of Model Land." 
#


#
# Welcome to the world of model-building workflows!!    
#

#
# In fact, the next task on this hw is to run at least one more ML workflow:   
#          (2) Digits, (ec) Your-own-data, ...
#


