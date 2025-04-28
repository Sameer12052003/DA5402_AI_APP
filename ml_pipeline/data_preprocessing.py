import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from collections import Counter

def preprocess(data_path):

    # Load the pandas dataframe 
    df = pd.read_csv(data_path) 

    # Separate features and target
    X = df.drop(columns=['Class','Time'])
    y = df['Class']

    # # Impute missing values (if any)
    # imputer = SimpleImputer(strategy='mean')
    # X_imputed = imputer.fit_transform(X)

    # # Feature scaling
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X_imputed)
    
    return X, y
