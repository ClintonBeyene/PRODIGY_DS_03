from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def model_selection(X_train, y_train):
    # Initialize the models 
    dt = DecisionTreeClassifier(random_state=42)
    rf = RandomForestClassifier(random_state=42)

    # Evaluate the model using cross-validation 
    scores_dt = cross_val_score(dt, X_train, y_train, cv=5)
    scores_rf = cross_val_score(rf, X_train, y_train, cv=5)

    # Calculate and print the mean cross-validation scores
    mean_score_dt = scores_dt.mean()
    mean_score_rf = scores_rf.mean()

    print("Decision Tree cross-validation score:", mean_score_dt)
    print("Random Forest cross-validation score:", mean_score_rf)

    return dt, rf
