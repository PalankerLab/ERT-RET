#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 12:46:48 2025

@author: Mohammad Asif Zaman

- Utility functions file. Import this in preprocessing, training and testing.
"""

# util_functions.py
import numpy as np

def Z_log(x):
    return np.log(x + 1e-6)

def Z_exp(x):
    return np.exp(x) - 1e-6
