#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:04:22 2025

@author: ab799
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the range for x in mm
x = np.linspace(-20, 20, 1000)  

# Define the exponential decay function (symmetric about x=0)
def exponential_decay(x, lam):
    return np.exp(-np.abs(x) * lam)  # Use absolute value to mirror decay on both sides

# Compute decay curves for λ = 0.15 and λ = 0.3
y1 = exponential_decay(x, 0.15) # Cortex
y2 = exponential_decay(x, 0.3) # Hippocampus

# Plotting
plt.figure(figsize=(6, 5))
plt.plot(x, y1, label='EDR Cortex', color='black')
plt.plot(x, y2, label='EDR Hippocampus', color='grey')

# Formatting
plt.xlabel('Distance / mm', fontsize=12)
plt.ylabel('Coupling strength', fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=12, direction='in')


#plt.title('Symmetric Exponential Decay: λ = 0.15 cm vs. 0.30 cm', fontsize=14)
plt.legend()
plt.xlim(-20, 20)
plt.ylim(0, 1)
plt.savefig(
    "/Users/ab799/Library/Mobile Documents/com~apple~CloudDocs/manuscript_HippoWaves/FigureItems/exponential_decay_plot.jpg", 
    format='jpg', bbox_inches='tight',dpi=300 )  # For SVG
plt.show()