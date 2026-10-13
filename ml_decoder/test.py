#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 4 18:43:19 2025

@author: Mohammad Asif Zaman

Sept. 17, 2025
 - Implemented bar plot option for experimental data

"""


import os
import torch as pt
import numpy as np
import pandas as pd
import matplotlib.pyplot as py
import joblib  # for loading Scaler
import torch.nn.functional as F
import matplotlib.cm as cm

from ml_decoder.util_fuctions import *

py.close('all')


# to recreate the published experimental results:
# LE (4-layer): scaling_scheme = 'a'
# RCS (3-layer): scaling_scheme = 'b'

# parameters
# =============================================================================

device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

Input_start_ind = 6   # which column to start from: 0-4 are resistivity columns, 5 is auto-potential
N_input = 17          # how many voltage measurements to use as input features
scaling_scheme = 'a'  # scaling methods: 'b' for RCS and 'a' for LE
bar_plot_data = 'n'   # put 'y' only for experimental data, not test data from comsol datasets. 
err_plot_data = 'y'
write_data = 'y'
y_pred_lim = 'y'
y_lim = 60

write_data_fname = 'decoded_res_4layer_' + str(N_input) + str(scaling_scheme) + '.csv'
# write_data_fname = 'decoded_res_1layer_' + str(N_input) + str(scaling_scheme) + '.csv'

# =============================================================================
print(f'Model input tensor length = {N_input}')
print(f'Scaling scheme = {scaling_scheme} \n')

# =============================================================================

# =============================================================================
# 4 layer retina datasets 33 spacing LE rats
script_dir = os.path.dirname(os.path.abspath(__file__))
data_train_dir = os.path.join(script_dir, 'data_4_layer_model/')
model_dir = os.path.join(script_dir, 'data_4_layer_model/')

# define the directory and file name of the test data (synthetic or the experimental trace to evaluate)
data_test_dir = os.path.join(script_dir, 'data_4_layer_model/')
fdata_name = 'test_33um_8000_samples.csv'
fdata_name = 'LE_average_trace_all_first_600um_ML_format.csv'

# =============================================================================

dict_model_path = model_dir + 'trained_dict_model_33_Nin = ' + str(N_input) +  str(scaling_scheme) + '.pt'
input_scaler_path = data_train_dir + 'input_scaler_pipeline_33_Nin = ' + str(N_input) +  str(scaling_scheme) + '.pkl'
target_scaler_path = data_train_dir + 'target_scaler_pipeline_33_Nin = ' + str(N_input) +  str(scaling_scheme) + '.pkl'

output_dim = 4

voltage_columns = np.arange(Input_start_ind, Input_start_ind + N_input)
resistivity_columns = np.arange(0,4)

# df = pd.read_csv(data_test_dir + fdata_name)
df = pd.read_csv(data_test_dir + fdata_name, engine='python', quoting=1, quotechar='"')

id_store = 'no id'

# spit input features (X) and target columns (y)
X_test = df.iloc[:, voltage_columns].values      # N_input voltage measurements
y_test = df.iloc[:, resistivity_columns].values  # 4 target resistivity values

# load scaler 
input_scaler = joblib.load(input_scaler_path)    # Scaler used during preprocessing
target_scaler = joblib.load(target_scaler_path)  # Scaler used during preprocessing

# Plot data before scaling
data_np = X_test
data_prescale_store = X_test
# Plot each row as a separate curve
for i in range(data_np.shape[0]):
    py.plot(data_np[i], label=f'Curve {i+1}')

# scale data
X_test = input_scaler.transform(X_test)    
X_test = pt.tensor(X_test, dtype=pt.float32)

# plot data after scaling
py.figure()
data_np = X_test
# Plot each row as a separate curve
for i in range(data_np.shape[0]):
    py.plot(data_np[i], label=f'Curve {i+1}')


# function definitions
def n_term_stat(x,n):
    ind = np.argsort(x)
    x_temp = x[ind]
    x_mean = []
    x_std = []
    for m in range(0,len(x)-n,n):
        x_mean.append( np.mean(x_temp[m:m+n]) )
        x_std.append ( np.std(x_temp[m:m+n]) )
        
    return np.array(x_mean), np.array(x_std)


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

# loading model
model = Simple_CNN(N_in=N_input).to(device)

# load trained model parameters
model.load_state_dict(pt.load(dict_model_path, map_location=device))

# set model to evaluation mode
model.eval()

# running model on test data
with pt.no_grad():
    y_pred = model(X_test.to(device)).cpu()

# inverse transform from the training scalar when needed
y_pred_np = target_scaler.inverse_transform(y_pred.numpy())
y_true_np = y_test

if y_pred_lim == 'y':
    y_pred_np[y_pred_np <=y_lim] = y_lim

# error calculation
epsilon = 1e-8
percent_error = np.abs((y_pred_np - y_true_np) / (y_true_np + epsilon)) * 100
abs_error = np.abs(y_pred_np - y_true_np) 

# total mean absolute error
# overall_mae = np.mean(np.abs(y_pred_np - y_true_np))
# print(f"Total MAE: {overall_mae:.4f}")

# error per resistivity channels
for m in range(output_dim):
    mae_m = np.mean(np.abs(y_pred_np[:, m] - y_true_np[:, m]))
    pae_m = 100*np.mean(np.abs(y_pred_np[:, m] - y_true_np[:, m])/(np.abs(y_true_np[:,m]) + epsilon) )
    pae_m = round(pae_m,2)
    print(f"rho_{m+1}: Avg. error = {mae_m:.4f} ohm.cm, % error = {pae_m:.2f}%")


# =============================================================================
# Plot error and comparison. For synthetic test data mainly.
# =============================================================================

if err_plot_data == 'y':
    
    py.figure(figsize=(8, 5))
    
    for m in range(output_dim):
        py.subplot(2,2,m+1)
        py.scatter(y_true_np[:, m], percent_error[:, m], alpha=0.6, s=12,  color = '#3c50b1')
        py.xlabel(f'True $\\rho_{{{m+1}}}$ ($\\Omega\\cdot\\mathrm{{cm}})$')
        py.ylabel(f'% error in $\\rho_{{{m+1}}}$ ')
        py.tight_layout()
        
    py.savefig(model_dir + 'percent error_csv.png') 
      
    py.figure(figsize=(8, 5))
    
    for m in range(output_dim):
        py.subplot(2,2,m+1)
        py.scatter(y_true_np[:, m], abs_error[:, m], alpha=0.6, s=12, color = '#495e35')
        py.xlabel(f'True $\\rho_{{{m+1}}}$ ($\\Omega\\cdot\\mathrm{{cm}})$')
        py.ylabel(f'Error in $\\rho_{{{m+1}}}$ ($\\Omega\\cdot\\mathrm{{cm}}$)')
        py.tight_layout()

    py.savefig(model_dir + 'total error_csv.png')

    py.figure(figsize=(8, 5))

    for m in range(output_dim):
        py.subplot(2,2,m+1)
        
        y_true_A, y_true_SD = n_term_stat(y_true_np[:,m], 10)
        y_pred_A, y_pred_SD = n_term_stat(y_pred_np[:,m], 10)
        
        err = 100*np.abs(y_true_np[:,m] - y_pred_np[:,m])/np.abs(y_true_np[:,m])

        p_err_A = 100*np.abs(y_true_A - y_pred_A)/np.abs(y_true_A) 
        p_err_SD = 100*np.abs(y_true_A - y_pred_A)/np.abs(y_true_A) 
        
        py.plot(y_true_A, p_err_A, alpha=1.0, color = '#495e35')
        py.plot(y_true_A, p_err_A + p_err_SD, alpha=0.5, color = '#495e35')
        py.plot(y_true_A, p_err_A - p_err_SD, alpha=0.5, color = '#495e35')
        py.fill_between(y_true_A, p_err_A - p_err_SD, p_err_A + p_err_SD, color='#495e35', alpha=0.3,  linewidth = 0)
    
        py.xlabel(f'True $\\rho_{{{m+1}}}$ ($\\Omega\\cdot\\mathrm{{cm}})$')
        py.ylabel(f'$\\rho_{{{m+1}}}$ error %')
        py.tight_layout()

    py.savefig(model_dir + 'error_csv.png')
# =============================================================================


# =============================================================================
# Bar plot features
# =============================================================================

# fname_prefix = ['1047','1050','1051','1053','1054','1056','1057'] # experimental data labels for bar plot legend
fname_prefix = ''
cmap = py.get_cmap('tab20c')  
colors = [cmap(m) for m in range(len(fname_prefix))]

if bar_plot_data == 'y':
    
    # Bar plot for comparing true and predicted values (Saline case mainly where true values are known)
    # =============================================================================
    df = pd.DataFrame({
    'Experimental resistivity values': y_true_np[:,0],
    'Predicted resistivity values': y_pred_np[:,0]
    })
    df.plot(kind='bar', alpha = 1, color=['#9fc8c8', '#298c8c'])
    py.xlabel('Sample no.')
    py.ylabel(f'$\\rho$ ($\\Omega\\cdot\\mathrm{{cm}})$')

    # =============================================================================
    
    # Bar plots for showing sample to sample variation
    # =============================================================================
    df = pd.DataFrame(y_pred_np.T, columns=[fname_prefix[m] for m in range(y_pred_np.shape[0])])
    df.index = ['NFL_GCL', 'IPL', 'Nu', 'PR']
    # Plot grouped bar chart
    df.plot(kind='bar', color = colors, figsize=(8, 5))
    py.ylabel(f'$\\rho$ ($\\Omega\\cdot\\mathrm{{cm}})$')
    py.legend(title='Sample ID')
    py.xticks(rotation=0)  # Make x-axis labels horizontal
    py.savefig(data_test_dir + 'decoded_res.svg')
# =============================================================================


# =============================================================================
# Export decoded resistivity data
# =============================================================================
if write_data == 'y':
    np.savetxt(data_test_dir + write_data_fname, y_pred_np, delimiter=',')
# =============================================================================




