#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:48:06 2023

@author: yigongqin
"""

import glob, re, dill
import matplotlib.pyplot as plt
rawdat_dir = '../FGR/'
graph_dir= './all/'
#files = glob.glob(rawdat_dir + 'seed*.pkl')
#graph_files = glob.glob(graph_dir + 'seed*.pkl')
qoi_dict = {'G':[], 'Rmax':[], 'span':[]}

for i in range(1443):
    files = glob.glob(rawdat_dir + '*seed'+str(i)+'_*.h5')
    graph_files = glob.glob(graph_dir + 'seed'+str(i)+'_*.pkl')
    if len(files)>0 and len(graph_files)>0:
        file = files[0]
        graph_file = graph_files[0]
    else:
        continue
    for qoi in qoi_dict:        
        if qoi == 'span':
            element = int(re.search(qoi+'(\d+)', graph_file).group(1))
        else:    
            element = float(re.search(qoi+'(\d+\.\d+)', file).group(1))
        qoi_dict[qoi].append(element)
        #print(element) 
#
print([len(i) for k, i in qoi_dict.items()])

fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(qoi_dict['G'], qoi_dict['Rmax'], s=qoi_dict['span'], c='k')
ax.set_xlabel(r'$G\ (K/\mu m)$')
ax.set_ylabel(r'$R\ (m/s) $')
plt.savefig('dz_grid.pdf',bbox_inches='tight')

qoi_dict['G_min'] = min(qoi_dict['G'])
qoi_dict['G_max'] = max(qoi_dict['G'])  

qoi_dict['R_min'] = min(qoi_dict['Rmax'])
qoi_dict['R_max'] = max(qoi_dict['Rmax'])  

qoi_dict['G'] = [(i-qoi_dict['G_min'])/(qoi_dict['G_max']-qoi_dict['G_min']) for i in qoi_dict['G']]
qoi_dict['R'] = [(i-qoi_dict['R_min'])/(qoi_dict['R_max']-qoi_dict['R_min']) for i in qoi_dict['Rmax']]

del qoi_dict['Rmax']

print(qoi_dict)

with open('GR_train_grid.pkl', 'wb') as outp:
    dill.dump(qoi_dict, outp)   
