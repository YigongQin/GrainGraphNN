#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 20:41:18 2021

@author: yigongqin
"""
from math import pi
import time
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from plot_funcs import plot_IO

import glob, sys,  copy
from input_data import assemb_data, device, todevice, tohost
from models import ConvLSTM_start, ConvLSTM_seq, y_norm, area_norm
from split_merge_reini import split_grain, merge_grain
from parameters import hyperparam, data_dir, valid_dir, model_dir
from utils import   divide_seq, divide_feat 

#torch.cuda.empty_cache()

mode = sys.argv[1]
all_id = int(sys.argv[2])-1


model_exist = False
if mode == 'test': model_exist = True
noPDE = True
plot_flag = True
skip_check = False




print('==========  GrainNN specification  =========')
print('2D grain microstructure evolution')
print('the mode is: ', mode, ', the model id is: ', all_id)
print('device: ',device)
print('model already exists, no training required: ', model_exist)
print('no PDE solver required, input is random: ', noPDE)
print('plot GrainNN verus PDE pointwise error: ', plot_flag)
print('\n')


hp = hyperparam(mode, all_id)
frames = hp.frames
G = hp.G
gap = int((hp.all_frames-1)/(frames-1))

print('************ setup model ***********')
print('==========  architecture  ========')
print('type -- s2s LSTM')

print('input window', hp.window,'; output window', hp.out_win)
print('epochs: ', hp.epoch, '; learning rate: ', hp.lr)
print('input feature dimension: ', hp.feature_dim)
print('hidden dim (layer size): ', hp.layer_size, '; number of layers', hp.layers)
print('convolution kernel size: ', hp.kernel_size)

seed_num = 35
torch.manual_seed(seed_num)
print('torch seed', seed_num)

if mode=='train' or mode == 'test': model = ConvLSTM_seq(hp, device)
if mode=='ini': model = ConvLSTM_start(hp, device)

model = model.double()
if device=='cuda':
  model.cuda()
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('total number of trained parameters ', pytorch_total_params)
print('\n')


print('************ setup data ***********')
datasets = sorted(glob.glob(data_dir))
testsets = sorted(glob.glob(valid_dir))


batch_size = 1
batch_train = len(datasets)
batch_test = len(testsets)
num_train = batch_size*batch_train
num_test = batch_size*batch_test


print('==========  data information  =========')
print('dataset dir: ',data_dir,'; batches: ', batch_train)
print('test dir: ', valid_dir,'; batches: ', batch_test)
print('number of train, test runs', num_train, num_test)
print('trust the data, skip check: ', skip_check)
print('data frames: ', hp.all_frames, '; GrainNN frames: ', frames, '; ratio: ', gap)
print('1d grid size (number of grains): ', G)
print('physical parameters: N_G orientations, e_k, G, R')
print('\n')




if mode == 'test':
    [G_list, R_list, e_list, Cl0, y0, input_] = assemb_data(num_test, batch_test, testsets, hp, mode, valid=True)
else:
    test_loader, [G_list, R_list, e_list, Cl0, y0, input_] = assemb_data(num_test, batch_test, testsets, hp, mode, valid=True)
    train_loader, _ = assemb_data(num_train, batch_train, datasets, hp, mode, valid=False)




'''
if skip_check == False:
 weird_sim = check_data_quality(frac_train, param_train, y_train, G, frames)
else: weird_sim=[]
print('nan', np.where(np.isnan(frac_train)))
weird_sim = np.array(weird_sim)[np.array(weird_sim)<num_train]
print('throw away simulations',weird_sim)
#### delete the data in the actual training fractions and parameters
#if len(weird_sim)>0:
# frac_train = np.delete(frac_train,weird_sim,0)
# param_train = np.delete(param_train,weird_sim,0)
#num_train -= len(weird_sim) 
print('actual num_train',num_train)
'''



# =====================================================================================


                               # TRAINING BLOCK


# =====================================================================================

def train(model, num_epochs, train_loader, test_loader):

    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),lr=hp.lr) 
                                 #weight_decay=1e-5) # <--

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1)


    train_loss = 0
    count = 0
    for  ix, (I_train, O_train, C_train) in enumerate(train_loader):   
        count += I_train.shape[0]
        recon, seq = model(I_train, C_train )
        train_loss += I_train.shape[0]*float(criterion(recon, O_train)) 
    train_loss/=count

    test_loss = 0
    count = 0
    for  ix, (I_test, O_test, C_test) in enumerate(test_loader):      
        count += I_test.shape[0]
        pred, seq = model(I_test, C_test )
        test_loss += I_test.shape[0]*float(criterion(pred, O_test)) 
    test_loss/=count

    print('Epoch:{}, Train loss:{:.6f}, valid loss:{:.6f}'.format(0, float(train_loss), float(test_loss)))
    train_list.append(float(train_loss))
    test_list.append(float(test_loss))  

    for epoch in range(num_epochs):


      if mode=='train' and epoch==num_epochs-10: optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
      train_loss = 0
      count = 0
      for  ix, (I_train, O_train, C_train) in enumerate(train_loader):   
         count += I_train.shape[0]
    
         recon, seq = model(I_train, C_train )
       
         loss = criterion(recon, O_train) 

         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
         
         train_loss += I_train.shape[0]*float(loss)
        # exit() 
      train_loss/=count
      test_loss = 0
      count = 0
      for  ix, (I_test, O_test, C_test) in enumerate(test_loader):

        count += I_test.shape[0]
        pred, seq = model(I_test, C_test)

        test_loss += I_test.shape[0]*float(criterion(pred, O_test)) 
 
      test_loss/=count
      print('Epoch:{}, Train loss:{:.6f}, valid loss:{:.6f}'.format(epoch+1, float(train_loss), float(test_loss)))
 
      train_list.append(float(loss))
      test_list.append(float(test_loss))       
      scheduler.step()

    return model 




if model_exist==False: 
  train_list=[]
  test_list=[]
  start = time.time()
  model=train(model, hp.epoch, train_loader, test_loader)
  end = time.time()
  print('training time',-start+end)
  if mode == 'train': torch.save(model.state_dict(), './lstmmodel'+str(all_id))
  if mode == 'ini': torch.save(model.state_dict(), './ini_lstmmodel'+str(all_id))
  fig, ax = plt.subplots() 
  ax.semilogy(train_list)
  ax.semilogy(test_list)
  txt = 'final train loss '+str('%1.2e'%train_list[-1])+' validation loss '+ str('%1.2e'%test_list[-1]) 
  fig.text(.5, .2, txt, ha='center')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend(['training loss','validation loss'])
  plt.title('training time:'+str( "%d"%int( (end-start)/60 ) )+'min')
  plt.savefig('mul_batch_loss.png')

if mode!='test' and model_exist==False: sio.savemat('loss_curve_mode'+mode+'.mat',{'train':train_list,'test':test_list})






# =====================================================================================


                           # INFERENCE BLOCK


# =====================================================================================






def network_inf(seq_out, model, ini_model, hp):
    if noPDE == False:
        seq_dat = seq_out[:,:hp.window,:,:]
    else: 


        seq_1 = seq_out[:,[0],:,:]   ## this can be generated randomly
        #print('sample', seq_1[0,0,:,:])
      

        seq_1_s, grainid_list, Cl_list= split_grain(seq_1, hp)
        
        seq_1_s[:,:,-1,:] = hp.dt

        output_model = ini_model(todevice(seq_1_s), todevice(Cl_list) )
        loss_feat = tohost( output_model[0] ) 
        frac = tohost(output_model[1])



        seq_out[:,1:hp.window,:4,:] = merge_grain(frac, loss_feat, hp, grainid_list, Cl_list)

        seq_dat = seq_out[:,:hp.window,:,:]

    if mode != 'ini':
              seq_dat[:,0,1,:] = seq_dat[:,1,1,:]
              seq_dat[:,0,3,:] = seq_dat[:,1,3,:] 




    #print('the sub simulations', grainid_list)
    alone = hp.pred_frames%hp.out_win
    pack = hp.pred_frames-alone

    for i in range(0,hp.pred_frames,hp.out_win):
        
        time_i = i
        if hp.dt*(time_i+hp.window+hp.out_win-1)>1: 
            time_i = int(1/hp.dt)-(hp.window+hp.out_win-1)
      

        seq_dat_s, grainid_list, Cl_list = split_grain( seq_dat, hp)

        seq_dat_s[:,:,-1,:] = (time_i+hp.window)*hp.dt ## the first output time
        print('nondim time: ', (time_i+hp.window)*hp.dt)

        output_model = model(todevice(seq_dat_s), todevice(Cl_list)  )
        loss_feat = tohost( output_model[0] ) 
        frac = tohost(output_model[1])


        if i>=pack and mode!='ini':
            seq_out[:,-alone:,:4,:] = merge_grain(frac[:,:alone,:], loss_feat[:,:alone,:], hp, grainid_list, Cl_list)
        else: 
            seq_out[:,hp.window+i:hp.window+i+hp.out_win,:4,:] = merge_grain(frac, loss_feat, hp, grainid_list, Cl_list)
        
        seq_dat = np.concatenate((seq_dat[:,hp.out_win:,:,:], seq_out[:,hp.window+i:hp.window+i+hp.out_win,:,:]),axis=1)
       

    frac_out, dfrac_out, area_out, dy_out = divide_feat(seq_out)
    frac_out *= hp.G_base/G
    dy_out = dy_out*y_norm
    dy_out[:,0] = 0
    y_out = np.cumsum(dy_out,axis=-1)+y0[:,np.newaxis]


    area_out = area_out*area_norm*hp.Cl[:,np.newaxis,np.newaxis]
    return frac_out, y_out, area_out


def ensemble(seq_out, inf_model_list):

    Nmodel = len(inf_model_list)


    frac_out = np.zeros((Nmodel,evolve_runs,frames,G)) ## final output
    area_out = np.zeros((Nmodel,evolve_runs,frames,G)) ## final output
    y_out = np.zeros((Nmodel,evolve_runs,frames))
    for i in range(Nmodel):
        print('\n')
        seq_i = copy.deepcopy(seq_out)
      #  param_i = copy.deepcopy(param_dat)
        all_id = inf_model_list[i]
        hp = hyperparam('test', all_id)
        hp.Cl = np.asarray(Cl0)
        model = ConvLSTM_seq(hp, device)
        model = model.double()
        if device=='cuda':
            model.cuda()
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model', inf_model_list[i], 'total number of trained parameters ', pytorch_total_params)
        model.load_state_dict(torch.load(model_dir+'/lstmmodel'+str(all_id)))
        model.eval()  


        ini_model = ConvLSTM_start(hp, device)
        ini_model = ini_model.double()
        if device=='cuda':
           ini_model.cuda()
        init_total_params = sum(p.numel() for p in ini_model.parameters() if p.requires_grad)
        print('model', inf_model_list[i], 'total number of trained parameters for initialize model', init_total_params)
        ini_model.load_state_dict(torch.load(model_dir+'/ini_lstmmodel'+str(all_id)))
        ini_model.eval()


        frac_out[i,:,:,:], y_out[i,:,:], area_out[i,:,:,:] = network_inf(seq_i, model, ini_model, hp)

 
    return np.mean(frac_out,axis=0), np.mean(y_out,axis=0), np.mean(area_out,axis=0)



evolve_runs = num_test #num_test

seq_out = np.zeros((evolve_runs,frames,hp.feature_dim,G))

seq_out[:,0,:,:] = input_[:,0,:,:]
seq_out[:,:,4:,:] = input_[:,:,4:,:]

if mode!='test':

    if model_exist:
      if mode == 'train' :
        model.load_state_dict(torch.load(model_dir+'/lstmmodel'+str(all_id)))
        model.eval()  
      if mode == 'ini':  
        model.load_state_dict(torch.load(model_dir+'/ini_lstmmodel'+str(all_id)))
        model.eval() 

    ini_model = ConvLSTM_start(hp, device)
    ini_model = ini_model.double()
    if device=='cuda':
       ini_model.cuda()
    init_total_params = sum(p.numel() for p in ini_model.parameters() if p.requires_grad)
    print('total number of trained parameters for initialize model', init_total_params)
    ini_model.load_state_dict(torch.load(model_dir+'/ini_lstmmodel'+str(all_id)))
    ini_model.eval()

    frac_out, y_out, area_out = network_inf(seq_out,  model, ini_model, hp)

if mode=='test':
    inf_model_list = hp.model_list
    nn_start = time.time()
    frac_out, y_out, area_out = ensemble(seq_out, inf_model_list)
    nn_end = time.time()
    print('===network inference time %f seconds =====', nn_end-nn_start)


'''
seq_reverse = copy.deepcopy(seq_out)
param_reverse = copy.deepcopy(param_dat)
seq_reverse [:,:,:G]      = np.flip(seq_out[:,:,:G],axis=-1)
seq_reverse [:,:,G:2*G]   = np.flip(seq_out[:,:,G:2*G],axis=-1)
seq_reverse [:,:,2*G:3*G] = np.flip(seq_out[:,:,2*G:3*G],axis=-1)

param_reverse [:,:G]      = np.flip(param_dat[:,:G],axis=-1)
param_reverse [:,G:2*G]   = -np.flip(param_dat[:,G:2*G],axis=-1)

print(seq_reverse[0,0,:],seq_out[0,0,:])
print(param_reverse[0,:],param_dat[0,:])

frac_out_r, y_out_r, area_out_r = network_inf(seq_reverse, param_reverse)
frac_out_r= np.flip(frac_out_r,axis=-1)
area_out_r= np.flip(area_out_r,axis=-1)


frac_out_f, y_out_f, area_out_f = network_inf(seq_out, param_dat)

frac_out = 0.5*(frac_out_f+frac_out_r)
y_out = 0.5*(y_out_f+y_out_r)
area_out = 0.5*(area_out_f+area_out_r)
'''

#darea_out[:,0,:] = 0
#area_out = np.cumsum(darea_out,axis=1)+area_all[num_train:num_train+evolve_runs,[0],:]
#print((y_out[0,:]))





# =====================================================================================


                           # TESTING BLOCK


# =====================================================================================






dice = np.zeros((num_test,G))
miss_rate_param = np.zeros(num_test)
run_per_param = int(evolve_runs/batch_test)
if run_per_param <1: run_per_param = 1

if mode == 'test': valid_train = True
else: valid = False
valid_train = True
if valid_train:
  aseq_test = np.arange(G)+1
  for batch_id in range(batch_test): 
   fname = testsets[batch_id] 
   f = h5py.File(fname, 'r')
   x = np.asarray(f['x_coordinates']) 
   y = np.asarray(f['y_coordinates'])
   xmin = x[1]; xmax = x[-2]
   ymin = y[1]; ymax = y[-2]
   print('xmin',xmin,'xmax',xmax,'ymin',ymin,'ymax',ymax)
   dx = x[1]-x[0] 
   fnx = len(x); fny = len(y); nx = fnx-2; ny = fny-2;
   print('nx,ny', nx,ny)

   angles_asse = np.asarray(f['angles'])
   alpha_asse = np.asarray(f['alpha'])
   G0 = float(G_list[batch_id]) 
   Rmax = float(R_list[batch_id]) 
   anis = float(e_list[batch_id])  
   for plot_idx in range( run_per_param ):  # in test dataset

     data_id = plot_idx*batch_test+batch_id
     #print('seq', param_test[data_id,:])
     pf_angles = angles_asse[plot_idx*(G+1):(plot_idx+1)*(G+1)]
     pf_angles[1:] = pf_angles[1:]*180/pi + 90     
     for dat_frames in [hp.all_frames-1]:
         
         inf_frames = dat_frames//gap + 1
         extra_time = dat_frames/gap - dat_frames//gap
         area = area_out[data_id,inf_frames-1,:]
         if extra_time>0: area += extra_time*(area_out[data_id,inf_frames,:]-area_out[data_id,inf_frames-1,:])

         frame_idx = hp.all_frames*plot_idx + dat_frames  ## uncomment this line if there are more than one frame in dat file
         frame_idx = plot_idx

         alpha_true = alpha_asse[frame_idx*fnx*fny:(frame_idx+1)*fnx*fny]
         alpha_true = np.reshape(alpha_true,(fnx,fny),order='F')[1:-1,1:-1]   

         miss_rate_param[data_id], dice[data_id,:] = plot_IO(anis,G0,Rmax,G,x,y,aseq_test,pf_angles,alpha_true,\
            y_out[data_id,:],frac_out[data_id,:,:].T, area, inf_frames, extra_time, plot_flag,data_id)

         print('realization id, param id', plot_idx, batch_id, 'frame id' , dat_frames, 'miss%',miss_rate_param[data_id])



#fig, ax = plt.subplots() 

x = np.array(e_list,dtype=float)
y = np.array(G_list,dtype=float)
z = np.array(R_list,dtype=float)
u = np.array(miss_rate_param,dtype=float)

print(x)
print(y)
print(z)
print(u)
print('for model ', all_id, 'the mean error', np.mean(u))

ave_err = np.mean(u)

if mode == 'test':
  print('all id', all_id, 'layer_size', hp.layer_size, 'learning_rate', hp.lr, \
    'num_layers', hp.layers, 'frames', frames, 'out win', hp.out_win, 'err', ave_err)
else:
      print('all id', all_id, 'layer_size', hp.layer_size, 'learning_rate', hp.lr, \
    'num_layers', hp.layers, 'frames', frames, 'out win', hp.out_win, 'err', ave_err, 'time', -start+end)
sio.savemat('2D_train'+str(num_train)+'_test'+str(num_test)+'_mode_'+mode+'_id_'+str(all_id)+'err'+str('%1.3f'%ave_err)+'.mat',{'frac_out':frac_out,'y_out':y_out,'area_out':area_out,'e':x,'G':y,'R':z,'err':u,'dice':dice,'Cl':Cl0,\
  'input':input_,'layer_size':hp.layer_size, 'learning_rate':hp.lr, 'num_layers':hp.layers, 'frames':frames}) #, \
  #'frac_true':frac_test,'y_true':y_test,'area_true':area_test})


