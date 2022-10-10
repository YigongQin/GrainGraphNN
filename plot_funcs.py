#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 16:52:09 2021

@author: yigongqin
"""
import numpy as np
import scipy.io as sio
import h5py
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.mathtext as mathtext
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time
from numba import njit, prange
var_list = ['Uc','phi','alpha']
range_l = [0,-5,0]
range_h = [5,5,90]
fid=2
var = var_list[fid]
vmin = np.float64(range_l[fid])
vmax = np.float64(range_h[fid])
ft=24
plt.rcParams.update({'font.size': ft})
#plt.style.use("dark_background")
mathtext.FontConstantsBase.sub1 = 0.2  
fg_color='white'; bg_color='black'


from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
coolwarm = cm.get_cmap('coolwarm', 256)
newcolors = coolwarm(np.linspace(0, 1, 256*100))
ly = np.array([255/256, 255/256, 210/256, 1])
newcolors[0, :] = ly
newcmp = ListedColormap(newcolors)

def subplot_rountine(fig, ax, cs, idx):
    

      if idx ==1: 
        ax.yaxis.set_ticks([0,30,60])
       # ax.yaxis.set_ticks([0,30,60,90,120,150])
        ax.set_xlabel(r'$x (\mu m)$')
        ax.set_ylabel(r'$y (\mu m)$')
      else: 
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
      ax.spines['bottom'].set_color(bg_color);ax.spines['left'].set_color(bg_color)
      ax.yaxis.label.set_color(bg_color); ax.xaxis.label.set_color(bg_color)
      ax.tick_params(axis='x', colors=bg_color); ax.tick_params(axis='y', colors=bg_color);
      if idx==1:
        axins = inset_axes(ax,width=0.25,height=2.5,loc='upper left')
        cbar = fig.colorbar(cs,cax = axins)#,ticks=[1, 2, 3,4,5])
        cbar.set_label(r'$\alpha_0$', color=bg_color)
        cbar.ax.yaxis.set_tick_params(color=bg_color)
        cbar.outline.set_edgecolor(bg_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=bg_color)
      if idx==4:
         cs.set_clim(0,1)
      else:
         cs.set_clim(vmin, vmax)
      return

@njit(parallel=True)
def reconstruction(field, temp_piece, y_range, area, G):
    y_f = y_range[-1]
    y_0 = y_range[0]
    ylen = len(y_range)
    
    for k in range(1):
      for g in prange(G):
        for j in prange(ylen):
              
              field[temp_piece[g, j+y_0]:temp_piece[g+1, j+y_0],j+y_0] = g+1

        top_layer = temp_piece[g+1, y_f]- temp_piece[g, y_f] 
        if  top_layer<5: height =0 
        else: height = int(np.round((area[g]/top_layer)))
        #print(height)
        field[temp_piece[g, y_f]:temp_piece[g+1, y_f], y_f:y_f+height] = g+1



def plot_IO(anis,G0,Rmax,G,x,y,aseq,pf_angles,alpha_true,tip_y,frac,area, final, extra_time, plot_flag, plot_idx):

    #print('angle sequence', aseq)
    #print(frac) 
    xmin = x[1]; xmax = x[-2]
    ymin = y[1]; ytop = y[-2]
    fnx = len(x); fny = len(y); nx = fnx-2; ny = fny-2;
    dx = x[1]-x[0]
    ntip_y = np.asarray(np.round(tip_y/dx),dtype=int)
    

    piece_len = np.asarray(np.round(np.cumsum(frac,axis=0)*nx),dtype=int)
    correction = piece_len[-1, :] - fnx
    for g in range(G//2, G):
      piece_len[g,:] -= correction


    field = np.zeros((nx,ny),dtype=int)
    ini_field = np.zeros((nx,ny),dtype=int)
    field[:,:ntip_y[0]+1] = alpha_true[:,:ntip_y[0]+1]
    ini_field[:,:ntip_y[0]+1] = alpha_true[:,:ntip_y[0]+1]

    temp_piece = np.zeros((G, ny), dtype=int)
    extra_y = 0
    if extra_time>0: extra_y = int(extra_time*( ntip_y[final] - ntip_y[final-1]))
    y_range = np.arange(ntip_y[0]+1, ntip_y[final-1]+extra_y+1)



#=========================start fill the final field=================
    start = time.time()
    for k in range(1):
      for g in range(G):
        fint = interp1d(ntip_y[:final], piece_len[g,:final],kind='linear')
        new_f = fint(y_range)
        temp_piece[g,y_range] = np.asarray(np.round(new_f),dtype=int)

    end = time.time()
    print(end-start)


    temp_piece = np.concatenate((0*temp_piece[:1,:], temp_piece),axis=0)  

    start = time.time()
    reconstruction(field, temp_piece, y_range, area, G)
    end = time.time()
    print(end-start)


    
    true_count = np.array([np.sum(alpha_true==g) for g in aseq])
    rom_count = np.array([np.sum(field==g) for g in aseq])
    inset = np.array([np.sum( (alpha_true==g)*(field==g) ) for g in aseq])
    dice = 2*inset/(true_count + rom_count)
    print('\n',dice)
#========================start plotting area, plot ini_field, alpha_true, and field======


    y_top = next( i for i,x  in  enumerate(np.mean(alpha_true, axis=0)) if x<1e-5)
    miss_rate = np.sum( alpha_true[:,:y_top]!=field[:,:y_top] )/(nx*y_top)
    miss = int(np.round(miss_rate*100))
    if plot_flag==True:
      fig = plt.figure(figsize=(7.5*xmax/20,12*ytop/80))
      txt = r'$\epsilon_k$'+str(anis)+'_G'+str("%1.1f"%G0)+r'_$R_{max}$'+str(Rmax)
     # fig.text(.5, .2, txt, ha='center')
      ax1 = fig.add_subplot(141)
      cs1 = ax1.imshow(pf_angles[ini_field].T,cmap=newcmp,origin='lower',extent= (xmin,xmax, ymin, ytop))
      subplot_rountine(fig, ax1, cs1, 1)
      #ax1.set_title('input:'+str(input_frac)+'%history',color=bg_color,fontsize=8)
      ax1.set_title('t=0',color=bg_color,fontsize=ft)
      ax2 = fig.add_subplot(142)
      cs2 = ax2.imshow(pf_angles[alpha_true].T,cmap=newcmp,origin='lower',extent= (xmin,xmax, ymin, ytop))
      subplot_rountine(fig, ax2, cs2, 2)
      ax2.set_title('PDE', color=bg_color,fontsize=ft)
      
      ax3 = fig.add_subplot(143)
      cs3 = ax3.imshow(pf_angles[field].T,cmap=newcmp,origin='lower',extent= (xmin,xmax, ymin, ytop))
      subplot_rountine(fig, ax3, cs3, 3)
      ax3.set_title('GrainNN', color=bg_color, fontsize=ft)
      
      ax4 = fig.add_subplot(144)
      cs4 = ax4.imshow(1*(alpha_true!=field).T,cmap='Reds',origin='lower',extent= (xmin,xmax, ymin, ytop))
      subplot_rountine(fig, ax4, cs4, 4)
      ax4.set_title('MR '+str(miss)+'%',color=bg_color,fontsize=ft)

      plt.savefig(var + '_grains' + str(G) + '_case' + str(plot_idx)+ '_frame' + str(final)+ '_anis' + str(anis)+'_G'+str("%1.1f"%G0)+'R' +str(Rmax) + '_error'+ str("%d"%miss) +'.pdf',dpi=600,facecolor="white", bbox_inches='tight')
      plt.close()

    
    return miss_rate, dice


def plot_synthetic(anis,G0,Rmax,G,x,y,aseq,tip_y_a, p_len_a, plot_idx,final,pf_angles, area_a, left_grains, nx_small):
    
    #print('angle sequence', aseq)
    #print(frac) 
    xmin = x[1]; xmax = x[-2]
    ymin = y[1]; ytop = y[-2]
    fnx = len(x); fny = len(y); nx = fnx-2; ny = fny-2;
    dx = x[1]-x[0]
    #nt=len(tip_y)
    #input_frac = int((window-1)/(nt-1)*100)

    #print(piece_len[-1,:])
    field = np.zeros((nx,ny),dtype=int)
    angle_field = np.zeros((nx,ny))



#=========================start fill the final field=================
    subruns = p_len_a.shape[0]
    for run in range(subruns):

      angles = pf_angles[run,:]
      tip_y = tip_y_a[run,:]
      ntip_y = np.asarray(tip_y/dx,dtype=int)

      p_len = p_len_a[run,:,:].T
      piece_len = np.cumsum(p_len,axis=0)
      correction = piece_len[-1, :] - nx_small
      for g in range(G//2, G):
        piece_len[g,:] -= correction

      piece0 = piece_len[:,0]

      if run ==0: rangeG = np.arange(G)[:3*G//4]
      elif run == subruns-1: rangeG=np.arange(G)[-3*G//4:]
      else: rangeG = np.arange(G)[G//4:3*G//4]

      temp_piece = np.zeros(G, dtype=int)
      miss=0
      for j in range(ntip_y[final-1]):
       #  loc = 0
         for g in range(G):
            if j <= ntip_y[0]: temp_piece[g] = piece0[g]
            else:
              fint = interp1d(ntip_y[:final], piece_len[g,:final],kind='linear')
              new_f = fint(j)
              temp_piece[g] = np.asarray(new_f,dtype=int)
         #print(temp_piece)
         #temp_piece = np.asarray(np.round(temp_piece/np.sum(temp_piece)*nx),dtype=int)
         for g in rangeG:
          if g==0:
            for i in range(left_grains[run], left_grains[run]+temp_piece[g]):
              if (i>nx-1 or j>ny-1): break

              angle_field[i,j] = angles[aseq[g]]

          else:
            for i in range(left_grains[run]+temp_piece[g-1], left_grains[run]+temp_piece[g]):
              if (i>nx-1 or j>ny-1): break
      
              angle_field[i,j] = angles[aseq[g]]



  #=========================start fill the extra field=================
      area = area_a[run]
      for g in rangeG:

        if p_len[g, final-1] ==0: height =0 
        else: height = int(area[g]/p_len[g, final-1])
        
        for j in range(ntip_y[final-1], ntip_y[final-1]+height):

          if g==0:
            for i in range(left_grains[run], left_grains[run]+temp_piece[g]):
              if (i>nx-1 or j>ny-1): break
              angle_field[i,j] = angles[aseq[g]]

          else:
            for i in range(left_grains[run]+temp_piece[g-1], left_grains[run]+temp_piece[g]):
              if (i>nx-1 or j>ny-1): break         
              angle_field[i,j] = angles[aseq[g]]

  #=========================fill in =================

    for j in range(ntip_y[final-1]):

       zeros = np.arange(nx)[angle_field[:,j]<1e-5]
       nonzeros = np.arange(nx)[angle_field[:,j]>=1e-5]

       fint = interp1d(nonzeros, angle_field[nonzeros,j],kind='nearest',fill_value='extrapolate')
       angle_field[zeros,j] = fint(zeros)

#========================start plotting area, plot ini_field, alpha_true, and field======


   # error=field-alpha_true[1:-1,1:-1]
    plot_flag=True
    if plot_flag==True:
      fig = plt.figure(figsize=(7.5*xmax/140,12*ytop/140))
      txt = r'$\epsilon_k$'+str(anis)+'_G'+str("%1.1f"%G0)+r'_$R_{max}$'+str(Rmax)
     # fig.text(.5, .2, txt, ha='center')
      
      ax3 = fig.add_subplot(111)
      cs3 = ax3.imshow(angle_field.T,cmap=newcmp,origin='lower',extent= (xmin,xmax, ymin, ytop))
      #subplot_rountine(fig, ax3, cs3, 3)
      ax3.yaxis.set_ticks([0,30,60])
      ax3.xaxis.set_ticks([0,50,100,150,200,250,300])
      ax3.set_xlabel(r'$x (\mu m)$')
      ax3.set_ylabel(r'$y (\mu m)$')
      cs3.set_clim(vmin, vmax)

      plt.savefig(var + '_grains' + str(G) + '_case' + str(plot_idx) + '_anis' + str(anis)+'_G'+str("%1.1f"%G0)+'R' +str(Rmax)+'.pdf',dpi=600,facecolor="white", bbox_inches='tight')
 
      plt.close()


