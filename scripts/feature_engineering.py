import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Function to apply label encoding
def label_encoding(df):
    columns = ['job', 'marital', 'education', 'default', 'housing', 
               'loan', 'contact', 'month', 'poutcome', 'y']
    for column in columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])

    return df

# Function to apply feature engineeing 
def feature_enginerring(df):
    # Create a new feature 'age_group' based on age
    df['age_group'] = pd.cut(df['age'], bins=[0, 20, 40, 60, 80], 
                             labels=['young', 'adult', 'middle_age', 'old'])
    # Convert 'age_group' to numerical variable using LabelEncoder
    le = LabelEncoder()
    df['age_group'] = le.fit_transform(df['age_group'])

    # Create a new feature 'contact_duration' based on 'duration'
    df['contach_duration'] = df['duration'] / 60

    return df