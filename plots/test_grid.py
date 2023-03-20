#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:49:10 2023

@author: yigongqin
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24})
G_, R_ = [], []

for seed in range(10000, 10100) :      
    np.random.seed(seed)
    G = np.random.random()*(10-0.5) + 0.5
    R = np.random.random()*(2-0.2) + 0.2
    G_.append(G)
    R_.append(R)
    
fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(G_, R_, c='k')
ax.set_xlabel(r'$G\ (K/\mu m)$')
ax.set_ylabel(r'$R\ (m/s) $')
plt.savefig('testGR.pdf',bbox_inches='tight')

