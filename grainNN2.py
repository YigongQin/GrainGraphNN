#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:35:27 2022

@author: yigongqin
"""
import argparse
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
from parameters import hyperparam
from utils import   divide_seq, divide_feat 

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




if __name__=='__main__':
    

    
    parser = argparse.ArgumentParser("Train the model.")
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--all_id", type=int, default=1)
    parser.add_argument("--model_exist", type=bool, default=False)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--model_dir ", type=str, default='./fecr_model/')
    parser.add_argument("--data_dir ", type=str, default='.')
    
    
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--plot_flag", type=bool, default=False)
    parser.add_argument("--noPDE", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=35)
    args = parser.parse_args()
    
    
    mode = args.mode
    all_id = args.all_id -1
    device = args.device
    
    if mode == 'test': args.model_exist = True
    
    
    
    
    print('==========  GrainNN specification  =========')
    print('3D grain microstructure evolution')
    print('the mode is: ', mode, ', the model id is: ', all_id)
    print('device: ', args.device)
    print('model already exists, no training required: ', args.model_exist)
    print('no PDE solver required, input is random: ', args.noPDE)
    print('plot GrainNN verus PDE pointwise error: ', args.plot_flag)
    print('\n')
    
    
    hp = hyperparam(mode, all_id)
    frames = hp.frames
    
    gap = int((hp.all_frames-1)/(frames-1))
    
    print('************ setup model ***********')
    print('==========  architecture  ========')
    print('type -- s2s LSTM')
    
    print('input window', hp.window,'; output window', hp.out_win)
    print('epochs: ', hp.epoch, '; learning rate: ', hp.lr)
    print('input feature dimension: ', hp.feature_dim)
    print('hidden dim (layer size): ', hp.layer_size, '; number of layers', hp.layers)
    print('convolution kernel size: ', hp.kernel_size)
    
    
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print('torch seed', seed)
    
    if mode=='train' or mode == 'test': model = ConvLSTM_seq(hp, device)
    if mode=='ini': model = ConvLSTM_start(hp, device)
    
    model = model.double()
    if device=='cuda':
        model.cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total number of trained parameters ', pytorch_total_params)
    print('\n')
    
    
    print('************ setup data ***********')
    datasets = sorted(glob.glob(args.data_dir))
    testsets = sorted(glob.glob(args.valid_dir))
    
    
    batch_size = 1
    batch_train = len(datasets)
    batch_test = len(testsets)
    num_train = batch_size*batch_train
    num_test = batch_size*batch_test
    
    
    print('==========  data information  =========')
    print('dataset dir: ',data_dir,'; batches: ', batch_train)
    print('test dir: ', valid_dir,'; batches: ', batch_test)
    print('number of train, test runs', num_train, num_test)
    print('data frames: ', hp.all_frames, '; GrainNN frames: ', frames, '; ratio: ', gap)
    print('1d grid size (number of grains): ',hp.G)
    print('physical parameters: N_G orientations, e_k, G, R')
    print('\n')
    
    
    
    
    if mode == 'test':
        [G_list, R_list, e_list, Cl0, y0, input_] = assemb_data(num_test, batch_test, testsets, hp, mode, valid=True)
    else:
        test_loader, [G_list, R_list, e_list, Cl0, y0, input_] = assemb_data(num_test, batch_test, testsets, hp, mode, valid=True)
        train_loader, _ = assemb_data(num_train, batch_train, datasets, hp, mode, valid=False)
        



    if args.model_exist==False: 
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
    
    if mode!='test' and args.model_exist==False: 
        sio.savemat('loss_curve_mode'+mode+'.mat',{'train':train_list,'test':test_list})




        
        
    evolve_runs = num_test #num_test
    
    seq_out = np.zeros((evolve_runs,frames,hp.feature_dim,hp.G))
    
    seq_out[:,0,:,:] = input_[:,0,:,:]
    seq_out[:,:,4:,:] = input_[:,:,4:,:]
    
    if mode!='test':
    
        if args.model_exist:
          if mode == 'train' :
            model.load_state_dict(torch.load(args.model_dir+'/lstmmodel'+str(all_id)))
            model.eval()  
          if mode == 'ini':  
            model.load_state_dict(torch.load(args.model_dir+'/ini_lstmmodel'+str(all_id)))
            model.eval() 
    
        ini_model = ConvLSTM_start(hp, device)
        ini_model = ini_model.double()
        if device=='cuda':
           ini_model.cuda()
        init_total_params = sum(p.numel() for p in ini_model.parameters() if p.requires_grad)
        print('total number of trained parameters for initialize model', init_total_params)
        ini_model.load_state_dict(torch.load(args.model_dir+'/ini_lstmmodel'+str(all_id)))
        ini_model.eval()
    
        frac_out, y_out, area_out = network_inf(seq_out,  model, ini_model, hp)
    
    if mode=='test':
        inf_model_list = hp.model_list
        nn_start = time.time()
        frac_out, y_out, area_out = ensemble(seq_out, inf_model_list)
        nn_end = time.time()
        print('===network inference time %f seconds =====', nn_end-nn_start)




    
