#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:38:43 2023

@author: yigongqin
"""

from graph_datastruct import graph
from math import pi
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
plt.rcParams.update({'font.size': 24})


g1 = graph(lxd = 400, seed = 1, BC = 'noflux', noise = 0.03) 
fig, ax = plt.subplots(1, 1, figsize=(20, 20), gridspec_kw={'width_ratios': [1], 'height_ratios': [1]})

    
ax.imshow(g1.theta_z[g1.alpha_field]/pi*180, origin='lower', cmap='coolwarm', vmin=0, vmax=90)
ax.set_xticks([])
ax.set_yticks([])

