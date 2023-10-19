#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 23:00:45 2023

@author: yigongqin
"""

def user_defined_config():
    
    config = {}
    
    config['meltpool'] = 'line'                               # line or cylinder

    config['boundary'] = 'noflux'                           # periodic or noflux
      
    config['geometry'] = {'lxd':40,                           # [um] length of domain 
                          'yx_asp_ratio':1,                   # lyd/lxd, must be <=1, assume lxd>=lyd is the longer dimension
                          'zx_asp_ratio':1.2,                 # lzd/lxd
                          'r0':1,                             # [um] radius for cylindrical meltpool
                          'z0':2,                             # [um] for line z0 is initial height of the interface 
                                                              #      for cylinder z0 is height of center of cylinder above lzd
                         }
    config['physical_parameters'] = {'G':1,                   # [K/um] temperature gradient
                                     'R':1,                   # [m/s] pulling velocity
                                    }
    config['initial_parameters'] = {'grain_size_mean':4,      # [um] 2 to 8 is tested, values out of this range accuracy is not guaranted 
                                    'mesh_size':0.08,         # [um] the mesh size between grid points in image
                                    'noise_level':0.01,       # noise added to hexagonal shaped grain crossection
                                    'seed':1,                 # realization of initial voronoi diagram
                                   }
    
    return config


"""
parameters used in training, not used in inference
patch_size = 40 um  # lxd used for training 
frames = 121        # frames sampled per simulation

"""