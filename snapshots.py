#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 18:08:00 2020

@author: yigongqin
"""
from math import pi
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import matplotlib.mathtext as mathtext
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import colors
import glob, os, re, argparse
plt.rcParams.update({'font.size': 10})
mathtext.FontConstantsBase.sub1 = 0.


def subplot_field(ax, i, j, seed, time):
    
    datasets = sorted((glob.glob(args.rawdat_dir + '/*seed'+str(seed)+'_*')))
    fname =datasets[0]#; print(fname)
    f = h5py.File(str(fname), 'r')
   # number_list=re.findall(r"[-+]?\d*\.\d+|\d+", fname)
   # R= float(number_list[8])
  #  G = float(number_list[7])
    
   # tid= 3*i*col+3*j
    
    alpha_id = (f[var])[time*length:(time+1)*length]
    angles = np.asarray(f['angles'])
  
    u = np.asarray(alpha_id).reshape((fnx,fny),order='F')[1:-1,1:-1]
   # u = np.rot90(u,3)
   # u = ( angles[u+Grains]/pi*180)*(u>0)
    
    #time = t0 #idx[j]/20*t0
    #if time == 0.0: plt.title('t = 0' + ' s',color=bg_color)
    #else: plt.title('t = '+str('%4.2e'%time) + ' s',color=bg_color)
    
    # cs = ax[i][j].imshow(u.T,cmap=plt.get_cmap('coolwarm_r'),origin='lower',extent= ( 0, lx, 0, ly))
    cs = ax[i][j].imshow(u.T,cmap=plt.get_cmap('coolwarm_r'),extent= ( 0, lx, 0, ly))
  #  ax[i].set_xlabel('('+case[i]+')'+' $G$'+str(int(G))+'_$R$'+str('%1.1f'%R)+'_$\epsilon_k$'+str('%1.2f'%anis)); 
    #ax[i].set_ylabel('$y\ (\mu m)$'); 
    ax[i][j].spines['bottom'].set_color(bg_color);ax[i][j].spines['left'].set_color(bg_color)
    ax[i][j].yaxis.label.set_color(bg_color); ax[i][j].xaxis.label.set_color(bg_color)
    ax[i][j].tick_params(axis='x', colors=bg_color); ax[i][j].tick_params(axis='y', colors=bg_color);
    ax[i][j].set_xticks([])
    ax[i][j].set_yticks([])
    """
    #plt.locator_params(nbins=3)
    if i==-10:
        axins = inset_axes(ax[i][j],width="3%",height="50%",loc='upper left')#,bbox_to_anchor=(1.05, 0., 1, 1),bbox_transform=ax[i].transax[i]es,borderpad=0,)
        cbar = fig.colorbar(cs,cax = axins)#,ticks=[1, 2, 3,4,5])
        cbar.set_label(r'$\alpha_0$', color=fg_color)
        cbar.ax.yaxis.set_tick_params(color=fg_color)
        cbar.outline.set_edgecolor(fg_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=fg_color)
        cs.set_clim(vmin, vmax)
        #plt.show()
    #plt.savefig(filebase + '_8grains_test_' +str(tid)+ '.pdf',dpi=800,facecolor="white", bbox_inches='tight')
    #plt.close()
   # print(u.shape)
   # print(u.T)
   """


if __name__ == '__main__':

    parser = argparse.ArgumentParser("plot snapshots")
    parser.add_argument("--mode", type=str, default = 'time')
    parser.add_argument("--frame", type=int, default = 25)
    parser.add_argument("--seed", type=int, default = 0)
    parser.add_argument("--runs", type=int, default = 1)
    parser.add_argument("--rawdat_dir", type=str, default = './')    
    args = parser.parse_args()
    
    datasets = sorted((glob.glob(args.rawdat_dir + '/*seed'+str(args.seed)+'_*')))
    filename = datasets[0]
    f0 = h5py.File(str(datasets[0]), 'r')
    
    number_list=re.findall(r"[-+]?\d*\.\d+|\d+", filename)
    
    
    Grains = int(number_list[0]); print('grains',Grains)
    frames = int(number_list[2])+1; print('frames',frames)
    
    f = h5py.File(filename, 'r')
    x = np.asarray(f['x_coordinates'])
    y = np.asarray(f['y_coordinates'])
    z = np.asarray(f['z_coordinates'])
    fnx = len(x); 
    fny = len(y);
    fnz = len(z)
    
    length = fnx*fny
    full_length = fnx*fny*fnz
    
    
    lx = 10
    ratio=1
    ly = lx*ratio
    print('the limits for geometry lx, ly: ',lx,ly)
    
    
    
    var_list = ['Uc','phi','cross_sec']
    range_l = [0,-1,0]
    range_h = [5,1,Grains]
    fid=2
    var = var_list[fid]
    vmin = np.float64(range_l[fid])
    vmax = np.float64(range_h[fid])
    print('the field variable: ',var,', range (limits):', vmin, vmax)
    fg_color='white'; bg_color='black'
    
    if args.mode == 'time':

        num_plots = args.frame
    elif args.mode == 'run':
        num_plots = args.runs
        
    row = int(np.sqrt(num_plots))
    row = row//5*5
    col = num_plots//row
    print('number of rows and columns', row, col)
    fig, ax = plt.subplots(row, col, figsize=(10,10))
    
    for i in range(row):
        for j in range(col):
            
            if args.mode == 'time': 
                seed = args.seed
                time = i*col+j
                subplot_field(ax, i, j, seed, time)
    
            elif args.mode == 'run':
                time = args.frame - 1
                seed = i*col+j
                subplot_field(ax, i, j, seed, time)
    
            else:
                raise ValueError
    fig.savefig('topview.png',dpi=600, bbox_inches='tight')








