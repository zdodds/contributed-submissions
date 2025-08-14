import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


digits_df = pd.read_csv("digits_cleaned.csv")


print("Columns in digits_df:", digits_df.columns)




TARGET_COLUMN = 'actual_digit'
X_all = digits_df.drop(TARGET_COLUMN, axis=1)
y_all = digits_df[TARGET_COLUMN]


SPECIES = [str(x) for x in digits_df[TARGET_COLUMN].unique()] 
print("SPECIES are\n", SPECIES)




X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)


param_grid = {'max_depth': range(1, 7)}  
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42),
                           param_grid,
                           cv=5, 
                           scoring='accuracy')
grid_search.fit(X_train, y_train)

best_max_depth = grid_search.best_params_['max_depth']
print(f"Best max_depth found by cross-validation: {best_max_depth}")


dt_model = DecisionTreeClassifier(max_depth=best_max_depth, random_state=42)
dt_model.fit(X_train, y_train)


dt_pred = dt_model.predict(X_test)
print("Decision Tree Model Evaluation:")
print(classification_report(y_test, dt_pred))
print("Accuracy:", accuracy_score(y_test, dt_pred))


dt_model_visual = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model_visual.fit(X_train, y_train)

plt.figure(figsize=(20,10)) 
plot_tree(dt_model_visual,
          feature_names=X_all.columns,
          class_names=SPECIES,
          filled=True)
plt.title("Decision Tree with max_depth=3")
plt.show()


param_grid_rf = {
    'n_estimators': [50, 100, 200],  
    'max_depth': range(3, 7)        
}


grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42),
                              param_grid_rf,
                              cv=3,  
                              scoring='accuracy')
grid_search_rf.fit(X_train, y_train)

best_rf_model = grid_search_rf.best_estimator_
print("Best Random Forest Model:", best_rf_model)


rf_pred = best_rf_model.predict(X_test)
print("\nRandom Forest Model Evaluation:")
print(classification_report(y_test, rf_pred))
print("Accuracy:", accuracy_score(y_test, rf_pred))


LoD = [[0,0,0,8,14,0,0,0,0,0,5,16,11,0,0,0,0,1,15,14,1,6,0,0,0,7,16,5,3,16,8,0,0,8,16,8,14,16,2,0,0,0,6,14,16,11,0,0,0,0,0,6,16,4,0,0,0,0,0,10,15,0,0,0],
[0,0,0,5,14,12,2,0,0,0,7,15,8,14,4,0,0,0,6,2,3,13,1,0,0,0,0,1,13,4,0,0,0,0,1,11,9,0,0,0,0,8,16,13,0,0,0,0,0,5,14,16,11,2,0,0,0,0,0,6,12,13,3,0],
[0,0,0,3,16,3,0,0,0,0,0,12,16,2,0,0,0,0,8,16,16,4,0,0,0,7,16,15,16,12,11,0,0,8,16,16,16,13,3,0,0,0,0,7,14,1,0,0,0,0,0,6,16,0,0,0,0,0,0,4,14,0,0,0],
[0,0,0,3,15,10,1,0,0,0,0,11,10,16,4,0,0,0,0,12,1,15,6,0,0,0,0,3,4,15,4,0,0,0,0,6,15,6,0,0,0,4,15,16,9,0,0,0,0,0,13,16,15,9,3,0,0,0,0,4,9,14,7,0],
[0,0,0,3,16,3,0,0,0,0,0,10,16,11,0,0,0,0,4,16,16,8,0,0,0,2,14,12,16,5,0,0,0,10,16,14,16,16,11,0,0,5,12,13,16,8,3,0,0,0,0,2,15,3,0,0,0,0,0,4,12,0,0,0],
[0,0,7,15,15,4,0,0,0,8,16,16,16,4,0,0,0,8,15,8,16,4,0,0,0,0,0,10,15,0,0,0,0,0,1,15,9,0,0,0,0,0,6,16,2,0,0,0,0,0,8,16,8,11,9,0,0,0,9,16,16,12,3,0]]


LoD_pred_dt = dt_model.predict(LoD)
LoD_pred_rf = best_rf_model.predict(LoD)

print("\nPredictions on LoD (Decision Tree):", LoD_pred_dt)
print("Predictions on LoD (Random Forest):", LoD_pred_rf)



feature_importances = best_rf_model.feature_importances_
print("\nFeature Importances from Random Forest:\n", feature_importances)


feature_importances_image = feature_importances.reshape(8, 8)


plt.figure(figsize=(8, 6))
sns.heatmap(feature_importances_image, annot=True, cmap="viridis", fmt=".3f")  # You can change the cmap
plt.title("Heatmap of Feature Importances")
plt.show()






#
# That's it!  Welcome to the world of model-building workflows!!    
#
#             Our prediction?  We'll be back for more ML! 
#


# If you'd like, the EC is to run a DT/RF workflow on your own data...   (in hw6ec_modeler.ipynb)




