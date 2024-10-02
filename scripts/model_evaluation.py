from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_evaluate(X_train, y_train, X_test, y_test, best_params_dt, best_params_rf):
    # Train a Decision Tree Classifier with the best hyperparameters
    clf_dt = DecisionTreeClassifier(max_depth=best_params_dt['max_depth'], 
                                    min_samples_split=best_params_dt['min_samples_split'], 
                                    random_state=42)
    clf_dt.fit(X_train, y_train)

    # Train a Random Forest Classifier with the best hyperparameters
    clf_rf = RandomForestClassifier(n_estimators=best_params_rf['n_estimators'], 
                                    max_depth=best_params_rf['max_depth'], 
                                    random_state=42)
    clf_rf.fit(X_train, y_train)

    # Evaluate the models
    y_pred_dt = clf_dt.predict(X_test)
    y_pred_rf = clf_rf.predict(X_test)

    print("Decision Tree Classifier Accuracy:", accuracy_score(y_test, y_pred_dt))
    print("Random Forest Classifier Accuracy:", accuracy_score(y_test, y_pred_rf))