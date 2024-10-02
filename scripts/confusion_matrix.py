from sklearn.metrics import classification_report, confusion_matrix

def evaluate_models(y_test, y_pred_dt, y_pred_rf):
    model_predictions = {
        "Decision Tree Classifier": y_pred_dt,
        "Random Forest Classifier": y_pred_rf
    }
    for model_name, y_pred in model_predictions.items():
        # Create a confusion matrix
        conf_mat = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix for {model_name}:\n", conf_mat)

        # Create a classification report
        class_report = classification_report(y_test, y_pred)
        print(f"Classification Report for {model_name}:\n", class_report)