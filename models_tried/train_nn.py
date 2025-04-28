import torch 
import torch.nn as nn  
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import classification_report
from data_preprocessing.data_preprocessing import preprocess
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

X_train_tensor = torch.tensor(X_train,dtype=torch.float32)
y_train_tensor = torch.tensor(y_train,dtype=torch.float32).unsqueeze(1)

X_val_tensor = torch.tensor(X_val,dtype=torch.float32)
y_val_tensor = torch.tensor(y_val,dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
val_dataset = TensorDataset(X_val_tensor,y_val_tensor)

train_loader = DataLoader(train_dataset,batch_size = 64,shuffle=True)

class NN(nn.Module):
    
    def __init__(self, input_dim):
        super(NN,self).__init__()
        
        self.layers = nn.Sequential(
            
            nn.Linear(input_dim,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        
        return self.layers(x)


# Initiliaze model

model = NN(input_dim=29)

criterion  = nn.BCELoss()

optimizer  = torch.optim.Adam(model.parameters(), lr = 0.001)   

# Train
t0 = time.time()

for epoch in range(5):
    
    for x,y in train_loader:
        
        pred = model(x) 
        
        loss = criterion(pred,y) 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1}/5, Loss: {loss.item()}")
    
tf = time.time()

model.eval()

with torch.no_grad():
    
    y_pred  = model(X_val_tensor) 
    
    y_pred = (y_pred > 0.8).int().numpy()
    
    
nn_report = classification_report(y_val,y_pred)

# save_report(nn_report,'nn_model','nn_report_report.txt')

print(nn_report)

print(f'Time taken: {tf-t0}')

     
        
    