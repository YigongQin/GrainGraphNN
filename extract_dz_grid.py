#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:48:06 2023

@author: yigongqin
"""

import glob, re
import matplotlib.pyplot as plt
rawdat_dir = '.'
files = glob.glob(rawdat_dir + '/seed*z50*')
qoi_dict = {'G':[], 'Rmax':[], 'span':[]}

for data_file in files:
    for qoi in qoi_dict:
        
        if qoi == 'span':
            element = int(re.search(qoi+'(\d+)', data_file).group(1))
        else:    
            element = float(re.search(qoi+'(\d+\.\d+)', data_file).group(1))
        qoi_dict[qoi].append(element)
        
print(qoi_dict)


fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(qoi_dict['G'], qoi_dict['Rmax'], s=qoi_dict['span'], c='k')
ax.set_xlabel(r'$G\ (K/\mu m)$')
ax.set_ylabel(r'$R\ (m/s) $')
plt.savefig('testGR.pdf',bbox_inches='tight')

