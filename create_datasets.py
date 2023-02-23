#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 10:26:38 2023

@author: yigongqin
"""

import glob, dill, random, sys, argparse


parser = argparse.ArgumentParser("Train the model.")
parser.add_argument("--data_dir", type=str, default='./quality/level2/')
parser.add_argument("--name", type=str, default='train')   
args = parser.parse_args()

datasets = sorted(glob.glob(args.data_dir + 'seed*'))
random.seed(35)
random.shuffle(datasets)
data_list = []

for case in datasets:
    with open(case, 'rb') as inp:  
        try:
            data_list = data_list + dill.load(inp)
        except:
            raise EOFError


with open('dataset_' + args.name + '.pkl', 'wb') as outp:
    dill.dump(data_list, outp)
