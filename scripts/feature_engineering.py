import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
import shap

# Function to apply label encoding
def label_encoding(df):
    columns = ['job', 'marital', 'education', 'default', 'housing', 
               'loan', 'contact', 'month', 'poutcome', 'y']
    for column in columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
