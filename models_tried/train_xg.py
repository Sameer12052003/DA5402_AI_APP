import os
import pandas as pd
from data_preprocessing.data_preprocessing import preprocess 
import sklearn 
import xgboost as xgb
import numpy as np
from sklearn.metrics import classification_report
import time
from imblearn.over_sampling import SMOTE
from collections import Counter

# Save classification report
def save_report(report_str, model_name, filename):
    with open(filename, "a") as f:
        f.write(f"\n=== {model_name} ===\n")
        f.write(report_str)
        f.write("\n" + "=" * 40 + "\n")


train_data_path  = 'dataset_splits/train.csv'
val_data_path  =  'dataset_splits/val.csv'

X_train, y_train = preprocess(train_data_path)
X_val, y_val = preprocess(val_data_path)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Apply SMOTE on training set
smote = SMOTE(random_state=42)
X_train, y_train_resampled = smote.fit_resample(X_train, y_train)

# Print shape and class balance before/after

print("Before SMOTE:", Counter(y_train))
print("After SMOTE: ", Counter(y_train_resampled))

y_train = y_train_resampled

model = xgb.XGBClassifier()

t0 = time.time()

model.fit(X_train,y_train)

tf = time.time()

y_pred  = model.predict(X_val)

xgb_report = classification_report(y_val,y_pred)

# save_report(xgb_report,'xgb_model','xgb_report_report.txt')

print(xgb_report)

print(f'Time taken: {tf-t0}')

