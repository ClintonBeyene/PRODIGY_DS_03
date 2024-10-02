import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def plot_roc_curves(X_test, y_test, models):

    for model_name, clf in models.items():
        # plot the ROC curves for both classifier in one figure
        plt.figure(figsize=(6,6))

        # Classifier predictions
        y_pred_proba = clf.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, color='blue', label=f'{model_name} Classifier')


        # Plot the diagonal line
        plt.plot([0, 1], [0, 1], color='red', line_style='--')

        # Add label and title 
        plt.title("ROC Curve For Decision Tree and Random Forest Classifiers")
        plt.xlabel("False Postitve Rate")
        plt.ylabel("True Postitve Rate")
        plt.legend()
    
    plt.show()
