import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def plot_roc_curves(X_test, y_test, models):
    plt.figure(figsize=(6,6))  # Create a single figure

    for model_name, clf in models.items():
        # Classifier predictions
        y_pred_proba = clf.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{model_name} Classifier')

    # Plot the diagonal line
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')

    # Add label and title 
    plt.title("ROC Curve For Decision Tree and Random Forest Classifiers")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

