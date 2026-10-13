#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 10:34:13 2025

@author: Mohammad Asif Zaman

Preprocessing the training data

Sept. 16, 2025
 - For each scaling scheme, the scalers are saved. But the input/output tensors are overwritten

"""

import os
import torch
import joblib
import numpy as np
import pandas as pd
from ml_decoder.util_fuctions import *
import sklearn.pipeline as pp
import sklearn.preprocessing as skp
from sklearn.model_selection import train_test_split

# to recreate the published experimental results:
# LE (4-layer): scaling_scheme = 'a', random_seed = 14
# RCS (3-layer): scaling_scheme = 'b', random_seed = 18

# define run parameters
Input_start_ind = 6   # which column to start from: 0-4 are resistivity columns, 5 is auto-potential
N_input = 17 		  # how many voltage measurements to use as input features
scaling_scheme = 'a'  # scaling methods: 'a', 'b', 'c', or 'd'
rnd_seed = 14
test_size = 0.001
validation_size = 0.1  # usually 0.1

script_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(script_dir, 'data_4_layer_model/')
fdata_name = 'train_4_layer_model_33um_30000_samples.csv'

input_scaler_fname = 'input_scaler_pipeline_33_Nin = ' + str(N_input) + str(scaling_scheme) + '.pkl'
target_scaler_fname = 'target_scaler_pipeline_33_Nin = ' + str(N_input) + str(scaling_scheme) + '.pkl'


# define columns that contain resistivity values and those that contain voltage values
resistivity_columns = np.arange(0,4)   # the first 5 columns are resistivity (4 layers + saline on top)
voltage_columns = np.arange(Input_start_ind,Input_start_ind + N_input)

log_transform = skp.FunctionTransformer(np.log1p, inverse_func=np.expm1)
log_transform_Z = skp.FunctionTransformer(Z_log, Z_exp)

# create a preprocessing pipeline:
if scaling_scheme == 'a':
    input_scaling_pipeline = pp.Pipeline([
        ('log', log_transform),
        ('scaler', skp.StandardScaler())
    ])
    
    target_scaling_pipeline = pp.Pipeline([
        ('log', log_transform),
        ('scaler', skp.StandardScaler())
    ])

if scaling_scheme == 'b':
    input_scaling_pipeline = pp.Pipeline([
        ('log', log_transform)
    ])
    
    target_scaling_pipeline = pp.Pipeline([
        ('scaler', skp.StandardScaler())
    ])

if scaling_scheme == 'c':
    input_scaling_pipeline = pp.Pipeline([
        ('logZ', log_transform_Z)
    ])
    
    target_scaling_pipeline = pp.Pipeline([
        ('scaler', skp.StandardScaler())
    ])

if scaling_scheme == 'd':
    input_scaling_pipeline = pp.Pipeline([
        ('logZ', log_transform_Z),
        ('scaler', skp.StandardScaler())
    ])
    
    target_scaling_pipeline = pp.Pipeline([
        ('logZ', log_transform_Z),
        ('scaler', skp.StandardScaler())
    ])

if scaling_scheme == 'e':
    input_scaling_pipeline = pp.Pipeline([
        ('scaler', skp.StandardScaler())
    ])
    
    target_scaling_pipeline = pp.Pipeline([
        ('scaler', skp.StandardScaler())
    ])

if scaling_scheme == 'f':
    input_scaling_pipeline = pp.Pipeline([
        ('log', log_transform),
        ('scaler', skp.StandardScaler())
    ])
    
    target_scaling_pipeline = pp.Pipeline([
        ('scaler', skp.StandardScaler())
    ])

# load the data from the given csv file
df = pd.read_csv(data_folder + fdata_name)

# split input features (voltages, X) and target columns (resistivities, y)
X = df.iloc[:, voltage_columns].values
y = df.iloc[:, resistivity_columns].values

# split data
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size= test_size, random_state=rnd_seed)
val_ratio = validation_size / (1 - test_size)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=rnd_seed)

X_train_scaled = input_scaling_pipeline.fit_transform(X_train) # fit only for the training data set
X_val_scaled = input_scaling_pipeline.transform(X_val)         # apply it for the validation data set
X_test_scaled = input_scaling_pipeline.transform(X_test)       # apply it for the test data set

y_train_scaled = target_scaling_pipeline.fit_transform(y_train) # fit only for the training data set
y_val_scaled = target_scaling_pipeline.transform(y_val)         # apply it for the validation data set
y_test_scaled = target_scaling_pipeline.transform(y_test)       # apply it for the test data set

# save scalars
joblib.dump(input_scaling_pipeline,  data_folder + input_scaler_fname)
joblib.dump(target_scaling_pipeline, data_folder + target_scaler_fname)

# convert data to tensor and save
def save_tensor(data, name):
    torch.save(torch.tensor(data, dtype=torch.float32), data_folder + f'{name}.pt')

save_tensor(X_train_scaled, 'X_train')
save_tensor(X_val_scaled, 'X_val')
save_tensor(X_test_scaled, 'X_test')
save_tensor(y_train_scaled, 'y_train')
save_tensor(y_val_scaled, 'y_val')
save_tensor(y_test_scaled, 'y_test')