#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:38:43 2023

@author: yigongqin
"""

import numpy as  np
from graph_datastruct import graph
from math import pi
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
plt.rcParams.update({'font.size': 24})


g1 = graph(lxd = 400, seed = 1, BC = 'noflux') 

alpha_top = np.load('alpha_top.npy')

"""

alpha_fields = [alpha_top, alpha_top, alpha_top.T]

align = [(200, 150), (200, 650), (650, 250)]


for i, (top, left) in enumerate(align):
    alpha = alpha_fields[i]
    size_x, size_y = alpha.shape
    g1.alpha_field[top:top+size_x, left:left+size_y] = alpha


fig1, ax1 = plt.subplots(1, 1, figsize=(20, 20), gridspec_kw={'width_ratios': [1], 'height_ratios': [1]})

    
ax1.imshow(g1.theta_z[g1.alpha_field]/pi*180,  cmap='coolwarm', vmin=0, vmax=90)
ax1.set_xticks([])
ax1.set_yticks([])

fig1.savefig('U' +'.png', dpi=400, bbox_inches='tight')


"""

alpha_fields = [alpha_top.T, alpha_top]

align = [(150, 250), (350, 400)]


for i, (top, left) in enumerate(align):
    alpha = alpha_fields[i]
    size_x, size_y = alpha.shape
    g1.alpha_field[top:top+size_x, left:left+size_y] = alpha


fig2, ax2 = plt.subplots(1, 1, figsize=(20, 20), gridspec_kw={'width_ratios': [1], 'height_ratios': [1]})

    
ax2.imshow(g1.theta_z[g1.alpha_field]/pi*180,  cmap='coolwarm', vmin=0, vmax=90)
ax2.set_xticks([])
ax2.set_yticks([])

fig2.savefig('T' +'.png', dpi=400, bbox_inches='tight')