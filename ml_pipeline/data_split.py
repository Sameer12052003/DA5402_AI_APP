import os
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

# Add to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Get the absolute path relative to this script
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Path to your dataset
original_dataset = os.path.join(base_dir,'credit_card_data/creditcard.csv')

# Directory where folders will be created
output_data_path = os.path.join(base_dir,'dataset_splits')   
os.makedirs(output_data_path,exist_ok=True)

# Split ratios
train_ratio = 0.6
val_ratio = 0.1
test_ratio = 0.3

# For reproducibility
random_state = 42               

# Load Data
df = pd.read_csv(original_dataset)

# Split into train + temp
train_df, temp_df = train_test_split(df, test_size = 1 - train_ratio, random_state=random_state)

# Split temp into val + test 
val_size = val_ratio / (val_ratio + test_ratio)
val_df, test_df = train_test_split(temp_df, test_size = 1 - val_size, random_state=random_state)

# Save splits
train_df.to_csv(os.path.join(output_data_path,  'train.csv'), index=False)
val_df.to_csv(os.path.join(output_data_path, 'val.csv'), index=False)
test_df.to_csv(os.path.join(base_dir,'shared_folder', 'test.csv'), index=False)

print("Dataset splitted and saved successfully!")
