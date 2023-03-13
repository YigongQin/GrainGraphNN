#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:49:30 2023

@author: yigongqin
"""

import h5py
import glob, re, os, argparse
import matplotlib.pyplot as plt
import numpy as np
from math import pi
plt.rcParams.update({'font.size': 24})

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("compare qoi curves")
    parser.add_argument("--rawdat_dir", type=str, default = './')
    parser.add_argument("--qoi", type=str, default = 'd')
    parser.add_argument("--frame", type=int, default = 121)
    args = parser.parse_args()
    
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    bins = 11 #np.arange(0, 20+1)
    upbound = 10
    files = sorted(glob.glob(args.rawdat_dir + '/*seed0*'))
    cnt = 0 
    for file in files:
        print(file)
        f = h5py.File(file, 'r')
        totalV_frames = np.asarray(f['total_area'])
        
        num_grains = int(re.search('grains(\d+)', file).group(1))
        totalV_frames = totalV_frames.reshape((num_grains, args.frame), order='F')  
        grain_volume = totalV_frames[:,-1].copy()
     #   scale_surface = np.sum(self.totalV_frames[:,time] - self.extraV_frames[:,time])/s**2/(self.final_height/self.mesh_size+1)
        
      #  self.grain_volume = self.grain_volume/scale_surface
        mesh_size = 0.08 if cnt==0 else 0.04
        cnt += 1
        grain_volume = grain_volume*mesh_size**3
        grain_size_t = np.cbrt(3*grain_volume/(4*pi))

        dis_t, _ =  np.histogram(grain_size_t, bins=bins, range=(0,upbound),density=True)
        d = np.arange(0, upbound+upbound/(bins-1), upbound/(bins-1))
        ax.plot(d, dis_t)
        
    ax.set_xlim(0, 12)
    ax.set_xlabel(r'$d\ (\mu m)$')
    ax.set_ylabel(r'$P$')
    ax.legend()  
    
    plt.savefig('size_dis.png', dpi=400, bbox_inches='tight')


