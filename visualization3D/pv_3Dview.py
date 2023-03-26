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
        self.base_width = 2
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
        self.theta_z[1:] = self.angles[len(self.angles)//2+1:]
        
        assert int(self.lxd) == int(self.x[-2])
        
        
        dx = self.x[1] - self.x[0]
        
        self.x /= self.lxd; self.y /= self.lxd; self.z /= self.lxd
        fnx, fny, fnz = len(self.x), len(self.y), len(self.z)
        print('grid ', fnx, fny, fnz)
        
        
        G = re.search('G(\d+)', self.data_file).group(1)
        R = re.search('Rmax(\d+)', self.data_file).group(1)
        data_frames = int(re.search('frames(\d+)', self.data_file).group(1))+1
        self.physical_params = {'G':G, 'R':R}
        print(self.physical_params)
        
        
        self.alpha_pde = np.asarray(f['alpha'])
        top_z = int(np.round(self.height/dx))
        
        self.alpha_pde = self.alpha_pde.reshape((fnx, fny,fnz),order='F')[1:-1,1:-1, 1:top_z]        
      #  self.alpha_pde[self.alpha_pde == 0] = np.nan
        self.alpha_pde = self.theta_z[self.alpha_pde]/pi*180
        
       # print(self.alpha_pde)
    
        grid = tvtk.ImageData(spacing=(dx, dx, dx), origin=(0, 0, 0), 
                              dimensions=self.alpha_pde.shape)
        
        grid.point_data.scalars = self.alpha_pde.ravel(order='F')
        grid.point_data.scalars.name = 'theta_z'
        
        
        
        self.dataname = rawdat_dir + 'seed'+str(self.seed) + '.vtk'
                   #rawdat_dir + 'seed'+str(self.seed)+'_G'+str('%2.2f'%self.physical_params['G'])\
                   #+'_R'+str('%2.2f'%self.physical_params['R'])+'.vtk'
        write_data(grid, self.dataname)
        

    def reconstruct(self, rawdat_dir: str = './', span: int = 6, alpha_field_list = None):
        
        self.data_file = (glob.glob(rawdat_dir + '/*seed'+str(self.seed)+'_*'))[0]
        f = h5py.File(self.data_file, 'r')
        self.x = np.asarray(f['x_coordinates'])
        self.y = np.asarray(f['y_coordinates'])
        self.z = np.asarray(f['z_coordinates']) 
        self.angles = np.asarray(f['angles']) 
        self.theta_z = np.zeros(1 + len(self.angles)//2)
        self.theta_z[1:] = self.angles[len(self.angles)//2+1:]
        
        assert int(self.lxd) == int(self.x[-2])
        
        
        dx = self.x[1] - self.x[0]
        
        self.x /= self.lxd; self.y /= self.lxd; self.z /= self.lxd
        fnx, fny, fnz = len(self.x), len(self.y), len(self.z)
        print('grid ', fnx, fny, fnz)
        
        
        G = re.search('G(\d+)', self.data_file).group(1)
        R = re.search('Rmax(\d+)', self.data_file).group(1)
        data_frames = int(re.search('frames(\d+)', self.data_file).group(1))+1
        self.physical_params = {'G':G, 'R':R}
        print(self.physical_params)
        

        
        if alpha_field_list:
            self.alpha_pde_frames = np.stack(alpha_field_list, axis=2)
        else:
            self.alpha_pde_frames = np.asarray(f['cross_sec'])
            self.alpha_pde_frames = self.alpha_pde_frames.reshape((fnx, fny, data_frames),order='F')[1:-1,1:-1,::span]               

        dx_frame = (50 - self.base_width)/(data_frames - 1)*span
        print('dx in plance ', dx,' between two planes',  dx_frame)
        

        
        top_z = int(np.round((self.height-self.base_width)/dx_frame)) + 1
        
        self.alpha_pde_frames = self.alpha_pde_frames[:, :, :top_z]     
        
        
      #  self.alpha_pde[self.alpha_pde == 0] = np.nan
        self.alpha_pde_frames = self.theta_z[self.alpha_pde_frames]/pi*180
        
        
        print('data shapes', self.alpha_pde_frames.shape)

    
        grid = tvtk.ImageData(spacing=(dx, dx, dx_frame), origin=(0, 0, 0), 
                              dimensions=self.alpha_pde_frames.shape)
        
        grid.point_data.scalars = self.alpha_pde_frames.ravel(order='F')
        grid.point_data.scalars.name = 'theta_z'
        
        
        
        self.dataname = rawdat_dir + 'seed'+str(self.seed) + 'leapz.vtk'

        write_data(grid, self.dataname)      
        
        
    def graph_recon(self, traj, rawdat_dir='./', span = 6, alpha_field_list = None):
        

 
        fnx, fny, fnz = len(traj.x), len(traj.y), len(traj.z)
        print('grid ', fnx, fny, fnz)
        dx = self.lxd/(fnx-3)

        data_frames = traj.frames

        self.alpha_pde_frames = np.stack(alpha_field_list, axis=2)
        layer_truth = traj.alpha_pde_frames[:, :, ::span] 
           

        dx_frame = (self.height - self.base_width)/(data_frames - 1)*span
        print('dx in plance ', dx,' between two planes',  dx_frame)
        
        
        top_z = int(np.round((50-self.base_width)/dx_frame)) + 1

        
        self.alpha_pde_frames = self.alpha_pde_frames[:, :, :top_z]     
        layer_truth = layer_truth[:, :, :top_z]

        err = 90*(self.alpha_pde_frames!=layer_truth)        
        
        
      #  self.alpha_pde[self.alpha_pde == 0] = np.nan
        self.alpha_pde_frames = traj.theta_z[self.alpha_pde_frames]/pi*180
        layer_truth = traj.theta_z[layer_truth]/pi*180
        
        print('data shapes', self.alpha_pde_frames.shape)

    
        grid = tvtk.ImageData(spacing=(dx, dx, dx_frame), origin=(0, 0, 0), 
                              dimensions=self.alpha_pde_frames.shape)
        
    
        grid.point_data.scalars = self.alpha_pde_frames.ravel(order='F')
        grid.point_data.scalars.name = 'theta_z'
        
        self.dataname = rawdat_dir + 'seed'+str(self.seed) + 'graph.vtk'
        write_data(grid, self.dataname) 


        grid.point_data.scalars = layer_truth.ravel(order='F') 
        grid.point_data.scalars.name = 'theta_z'
        
        self.dataname = rawdat_dir + 'seed'+str(self.seed) + 'leapz.vtk'
        write_data(grid, self.dataname) 

       # grid.point_data.scalars = err.ravel(order='F')  
       # grid.point_data.scalars.name = 'theta_z'
        
       # self.dataname = rawdat_dir + 'seed'+str(self.seed) + 'err.vtk'
       # write_data(grid, self.dataname) 

if __name__ == '__main__':


    parser = argparse.ArgumentParser("create 3D grain plots with paraview")

    parser.add_argument("--rawdat_dir", type=str, default = './')
    parser.add_argument("--pvpython_dir", type=str, default = '')
    parser.add_argument("--mode", type=str, default='truth')
    
    parser.add_argument("--seed", type=int, default = 20)
    parser.add_argument("--lxd", type=int, default = 40)
    parser.add_argument("--height", type=int, default = 50)
 
    args = parser.parse_args()
        
   # Gv = grain_visual(lxd = 20, seed = args.seed, height=20)   
    Gv = grain_visual(lxd=args.lxd, seed=args.seed, height=args.height)  
    if args.mode == 'truth':
        
        Gv.load(rawdat_dir=args.rawdat_dir)  
        
    elif args.mode == 'reconstruct':
        Gv.reconstruct(rawdat_dir=args.rawdat_dir)
    
    
    else:
        raise KeyError
   # args.pvpython_dir = '/Applications/ParaView-5.10.1.app/Contents/bin/'

        
        
"""
module load qt5 swr oneapi_rk paraview
"""
        
        
''' stack the 2 um height '''
   
#  stack_thick = int(self.base_width/dx_frame)
  
#  print('stack layers ', stack_thick)
  
#  base = np.tile(self.alpha_pde_frames[:,:,:1], (1,1,stack_thick))
#   self.alpha_pde_frames = np.concatenate([base, self.alpha_pde_frames], axis=-1)
