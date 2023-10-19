#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 23:00:45 2023

@author: yigongqin
"""

def user_defined_config():
    
    config = {'meltpool':'line',                    # line or cylinder
              'boundary':'periodic',                # periodic or noflux
             }
    geometry_descriptors = {'yx_asp_ratio':1,       # lyd/lxd, must be <=1, assume lxd>=lyd is the longer dimension
                            'zx_asp_ratio':1.2,     # lzd/lxd
                            'r0':1,                 # [um] radius for cylindrical meltpool
                            'z0':2,                 # for line: initial height of the interface 
                                                    # for cylinder: center of cylinder above lzd
                           }
    physical_parameters = {'G':1,                   # [K/um] temperature gradient
                           'R':1,                   # [m/s] pulling velocity
                          }
    initial_parameters = {'grain_size_mean':4,      # [um] 
                          'noise_level':0.01,       # noise added to hexagonal shaped grain crossection
                          'seed':1,                 # realization of initial voronoi diagram
                         }
    
    return config, geometry_descriptors, physical_parameters, initial_parameters