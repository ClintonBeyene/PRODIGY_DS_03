from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def hyperparameter_tuning(X_train, y_train):
    # Define the hyperparameter space for Decision Tree classifier
    param_grid_dt = {'max_depth': [3, 5, 10, 15],'min_samples_split': [2, 5, 10]}

    # Perform hyperparameter tuning using GridSearchCV for Decision Tree Classifier
    grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5)
    grid_dt.fit(X_train, y_train)

    # Print the best hyperparameters and the corresponding accuracy for Decision Tree Classifier
    print('Best Hyperparameters for Decision Tree Classifier:', grid_dt.best_params_)
    print('Best Accuracy for Decision Tree Classifier:', grid_dt.best_score_)

    # Define the hyperparameter space for Random Forest Classifier
    param_grid_rf = {'n_estimators': [10, 50, 100, 200], 'max_depth': [3, 5, 10, 15]}

    # Perform hyperparameter tuning using GridSearchCV for Random Forest Classifier
    grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5)
    grid_rf.fit(X_train, y_train)

    # Print the best hyperparameters and the corresponding accuracy for Random Forest Classifier
    print('Best Hyperparameters for Random Forest Classifier:', grid_rf.best_params_)
    print('Best Accuracy for Random Forest Classifier:', grid_rf.best_score_)

    return grid_dt.best_params_, grid_rf.best_params_