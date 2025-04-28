import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from collections import Counter

# Fit once for consistent scaling/imputing (optional for live use)
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

def preprocess(df):
    df = df.copy()  # Ensure original DataFrame is untouched

    # If 'Class' column is not in df (like during prediction), add dummy target
    # if 'Class' not in df.columns:
    #     df['Class'] = 0

    # Drop 'Time' if present
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])

    # Separate features and target
    X = df.drop(columns=['Class'])
    y = df['Class']

    # # Impute missing values
    # X_imputed = imputer.fit_transform(X)

    # # Scale features
    # X_scaled = scaler.fit_transform(X_imputed)

    return X, y
