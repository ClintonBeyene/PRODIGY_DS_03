import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importance(model, X):
    # Get the feature importances
    feature_importances_rf = model.feature_importances_

    # Sort the features by importance
    indices = np.argsort(feature_importances_rf)[::-1]
    sorted_features = X.columns[indices]
    sorted_importances = feature_importances_rf[indices]

    # Plot the feature importances 
    plt.figure(figsize=(10, 8))
    plt.bar(sorted_features, sorted_importances, color='skyblue')
    plt.xlabel('Features')
    plt.ylabel('Importances')
    plt.title('Feature Importances')
    plt.xticks(rotation=90)
    plt.show()