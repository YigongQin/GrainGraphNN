#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:49:10 2023

@author: yigongqin
"""

import glob, re

rawdat_dir = '.'
files = glob.glob(rawdat_dir + '/seed*z50*')
qoi_dict = {'elimp':[], '_t':[]}

for data_file in files:
    for qoi in qoi_dict:
        element = int(re.search(qoi+'\d+)', data_file).group(1))
        qoi_dict[qoi].append(element)
        
print(qoi_dict)
print(sum(qoi_dict['elimp']))