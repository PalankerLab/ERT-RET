#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 16:07:54 2025

@author: Mohammad Asif Zaman
"""

import os
import numpy as np
import torch as pt
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as py
import torch.nn.functional as F
import joblib  # for loading Scaler

from ml_decoder.util_fuctions import *


py.close('all')

# =============================================================================
# parameters
# =============================================================================

# Input tensor size
Input_start_ind = 6   # which column to start from: 0-4 are resistivity columns, 5 is auto-potential
N_input = 17          # how many voltage measurements to use as input features
scaling_scheme = 'b'  # scaling methods
output_dim = 4
batch_size = 64       # usually 64
epochs = 1000         # usually 1000
learning_rate = 1e-3  # usually 1e-3

# learning rate scheduling parameters
opt_fct = 0.9   # reduce by this factor every time it triggers 
ptn = 10        # wait for 10 iterations/epochs, usually 10

device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
# =============================================================================


# =============================================================================
# paths
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data_4_layer_model/')
model_dir = os.path.join(script_dir, 'data_4_layer_model/')

dict_model_path = model_dir + 'trained_dict_model_33_Nin = ' + str(N_input) +  str(scaling_scheme) + '.pt'
input_scaler_path = data_dir + 'input_scaler_pipeline_33_Nin = ' + str(N_input) +  str(scaling_scheme) + '.pkl'
target_scaler_path = data_dir + 'target_scaler_pipeline_33_Nin = ' + str(N_input) +  str(scaling_scheme) + '.pkl'

# =============================================================================

# =============================================================================
# Load data

# load training and validation data. Note that test data set is not loaded. This way, the model has no
# way of getting corrupted with data leakage.
# =============================================================================

x_train = pt.load(data_dir + 'X_train.pt')
y_train = pt.load(data_dir + 'y_train.pt')
x_val = pt.load(data_dir + 'X_val.pt')
y_val = pt.load(data_dir + 'y_val.pt')

train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# load scaler (not needed for processing. Only using it for writing log file)
input_scaler = joblib.load(input_scaler_path)  # Scaler used during preprocessing
target_scaler = joblib.load(target_scaler_path)  # Scaler used during preprocessing
# =============================================================================



# =============================================================================
# NN model define
# =============================================================================

class Simple_CNN(pt.nn.Module):
    def __init__(self, N_in):
        super().__init__()
        self.N_in = N_in

        # Conv1D layer
        self.conv1 = pt.nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)

        # Compute output length after Conv1D
        conv_out_len = N_in - self.conv1.kernel_size[0] + 1

        # Fully connected layers
        self.fc1 = pt.nn.Linear(64 * conv_out_len, 128)
        self.fc2 = pt.nn.Linear(128, 64)
        self.fc3 = pt.nn.Linear(64, 4)

    def forward(self, x):
        # x shape: (B, N_in) to reshape to (B, 1, N_in)
        x = x.unsqueeze(1)

        # Conv1D + ReLU
        x = F.relu(self.conv1(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# =============================================================================
# Model selection
# =============================================================================

model = Simple_CNN(N_in=N_input).to(device)
criterion = pt.nn.MSELoss()
optimizer = pt.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=opt_fct, patience=ptn
)
# =============================================================================



# =============================================================================
# training loop
# =============================================================================
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x_batch.size(0)

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # validation
    model.eval()
    val_loss = 0.0
    with pt.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            val_loss += loss.item() * x_batch.size(0)

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Iteration {epoch+1:04}/{epochs:04} | Train Loss: {train_loss:.3e} | Validation Loss: {val_loss:.3e} | LR: {current_lr:.3e}")

print("-----------------------")
print("-----------------------")

# save model 
pt.save(model.state_dict(),  dict_model_path)
print("\nModel Saved")
print("-----------------------")

# =============================================================================


# =============================================================================
# Plotting convergence plot
# =============================================================================

py.figure(figsize=(8, 5))
py.semilogy(np.linspace(0,epochs-1, epochs), train_losses, label='Train Loss')
py.semilogy(np.linspace(0,epochs-1, epochs), val_losses, label='Validation Loss')
py.xlabel('Epoch')
py.ylabel('MSE Loss (log scale)')
py.title('Training and Validation Error Over Epochs')
py.legend()
py.savefig(model_dir + 'Convergence plot_Nin = ' + str(N_input) + '_' + str(scaling_scheme) + '.png', dpi = 300)

# =============================================================================


# =============================================================================
# Export info
# =============================================================================

model_info = {
    "Model": str(model),
    "input_scaler": str(input_scaler),
    "target_scaler": str(target_scaler),
    "data_dir": data_dir,
    "model_dir": model_dir,
    "dict_model_path": dict_model_path,
    "batch_size": batch_size,
    "initial learning rate": learning_rate,
    "opt_fct": opt_fct,
    "patience, ptn": ptn,
    
}

with open(model_dir + 'log' + '_Nin = ' + str(N_input) + str(scaling_scheme) + '.txt', "w") as f:
    for key, value in model_info.items():
        f.write(f"{key}: {value}\n")

# =============================================================================

