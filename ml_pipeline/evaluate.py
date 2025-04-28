import joblib
import numpy as np
from sklearn.metrics import classification_report
from data_preprocessing import preprocess
import os,sys

# Add to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Get the absolute path relative to this script
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# Load data and model
X_val, y_val = preprocess(os.path.join(base_dir, 'dataset_splits/val.csv'))
loaded = joblib.load(os.path.join(base_dir,'shared_folder/model.pkl'))

model = loaded['model']
threshold = loaded['threshold']

# Predict fraud probabilities
y_proba = model.predict_proba(X_val)[:, 1]

# Apply a lower threshold (important!!)
y_pred = (y_proba > threshold).astype(int)
report = classification_report(y_val, y_pred)

print(report)

# Save report
with open(os.path.join(base_dir,"shared_folder/classification_report.txt"), "w") as f:
    f.write(report)
