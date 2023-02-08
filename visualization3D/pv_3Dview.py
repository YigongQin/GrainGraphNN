#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:37:04 2023

@author: yigongqin
"""

import h5py, glob, re, os, argparse
from  math import pi
import numpy as np
from tvtk.api import tvtk, write_data




class grain_visual:
    def __init__(self, 
                 lxd: float = 40, 
                 seed: int = 1, 
                 frames: int = 1, 
                 height: int = 50,
                 physical_params = {}):   

        self.lxd = lxd
        self.seed = seed
        self.height = height
        self.frames = frames # note that frames include the initial condition
        self.physical_params = physical_params
        

    def load(self, rawdat_dir: str = './'):
       
        
        self.data_file = (glob.glob(rawdat_dir + '/*seed'+str(self.seed)+'_*'))[0]
        f = h5py.File(self.data_file, 'r')
        self.x = np.asarray(f['x_coordinates'])
        self.y = np.asarray(f['y_coordinates'])
        self.z = np.asarray(f['z_coordinates']) 
        self.angles = np.asarray(f['angles']) 
        self.theta_z = np.zeros(1 + len(self.angles)//2)
        self.theta_z[1:] = self.angles[len(self.angles)//2:-1]
        
        assert int(self.lxd) == int(self.x[-2])
        
        
        dx = self.x[1] - self.x[0]
        
        self.x /= self.lxd; self.y /= self.lxd; self.z /= self.lxd
        fnx, fny, fnz = len(self.x), len(self.y), len(self.z)
        print('grid ', fnx, fny, fnz)
        
        
        number_list=re.findall(r"[-+]?\d*\.\d+|\d+", self.data_file)
        data_frames = int(number_list[2])+1
        
        self.physical_params = {'G':float(number_list[3]), 'R':float(number_list[4])}
        self.alpha_pde = np.asarray(f['alpha'])
        top_z = int(self.height/dx)
        
        self.alpha_pde = self.alpha_pde.reshape((fnx, fny,fnz),order='F')[1:-1,1:-1, 1:top_z]        
      #  self.alpha_pde[self.alpha_pde == 0] = np.nan
        self.alpha_pde = self.theta_z[self.alpha_pde]/pi*180
        
       # print(self.alpha_pde)
    
        grid = tvtk.ImageData(spacing=(dx, dx, dx), origin=(0, 0, 0), 
                              dimensions=self.alpha_pde.shape)
        
        grid.point_data.scalars = self.alpha_pde.ravel(order='F')
        grid.point_data.scalars.name = 'theta_z'
        
        
        print(self.physical_params)
        write_data(grid, 'test.vtk')
        

if __name__ == '__main__':


    parser = argparse.ArgumentParser("create 3D grain plots with paraview")

    parser.add_argument("--rawdat_dir", type=str, default = '../')
    parser.add_argument("--pvpython_dir", type=str, default = '')
    
    parser.add_argument("--seed", type=int, default = 20)
    parser.add_argument("--lxd", type=int, default = 40)
    parser.add_argument("--height", type=int, default = 50)
 
    args = parser.parse_args()
        
   # Gv = grain_visual(lxd = 20, seed = args.seed, height=20)   
    Gv = grain_visual(seed = args.seed)  
    Gv.load(rawdat_dir=args.rawdat_dir)   
    args.pvpython_dir = '/Applications/ParaView-5.11.0.app/Contents/bin/'
    os.system(args.pvpython_dir+'pvpython grain.py test.vtk ./ grain')       
        
        
        
"""
module load qt5 swr oneapi_rk paraview
"""
        
        
        