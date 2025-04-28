import os
import yaml
import logging
import subprocess
import sys

# Add to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Get the absolute path relative to this script
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

data_dvc_path = os.path.join(base_dir,"credit_card_data/creditcard.csv")

subprocess.run(["dvc", "pull", data_dvc_path], check=True)

