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

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("compare qoi curves")
    parser.add_argument("--rawdat_dir", type=str, default = './')
    parser.add_argument("--qoi", type=str, default = 'd')
    parser.add_argument("--frame", type=int, default = 121)
    args = parser.parse_args()
    
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    bins = np.arange(10+1)
    files = glob.glob(args.rawdat_dir + '/*seed0*')
    
    for file in files:
        print(file)
        f = h5py.File(file, 'r')
        totalV_frames = np.asarray(f['total_area'])
        
        num_grains = re.search('grains(\d+\.\d+)', file).group(1)
        totalV_frames = totalV_frames.reshape((num_grains, args.frame), order='F')  
        grain_volume = totalV_frames[:,-1].copy()
     #   scale_surface = np.sum(self.totalV_frames[:,time] - self.extraV_frames[:,time])/s**2/(self.final_height/self.mesh_size+1)
        
      #  self.grain_volume = self.grain_volume/scale_surface
        mesh_size = 0.08
        grain_volume = grain_volume*mesh_size**3
        grain_size_t = np.cbrt(3*grain_volume/(4*pi))

        dis_t, _ =  np.histogram(grain_size_t, bins, density=True)
    
        ax.plot(dis_t)
        
    ax.set_xlim(0, 10)
    ax.set_xlabel(r'$d\ (\mu m)$')
    ax.set_ylabel(r'$P$')
    ax.legend()  
    
    plt.savefig('size_dis.png', dpi=400, bbox_inches='tight')


