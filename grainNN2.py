#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:35:27 2022

@author: yigongqin
"""
import argparse
import time
import glob, dill
import random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import DynamicHeteroGraphTemporalSignal
from models import GrainNN2
from parameters import hyperparam
#from graph_datastruct import graph


def criterion(data, pred):
    
    return torch.mean((data['grain'] - pred['grain'])**2) \
         + torch.mean((data['joint'] - pred['joint'])**2)
    

def train(model, num_epochs, train_loader, test_loader):

    model.train()

    optimizer = torch.optim.Adam(model.parameters(),lr=hp.lr) 
                                 #weight_decay=1e-5) # <--

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1)


    train_loss, count = 0, 0

    for data in train_loader:  
        train.x = data.x_dict
        train.edge = data.edge_index_dict
        train.y = data.y_dict
        count += 1 #data.batch
        pred = model(data.x_dict, data.edge_index_dict)
        train_loss += float(criterion(data.y_dict, pred)) 
    train_loss/=count

    test_loss, count = 0, 0
    for data in test_loader:      
        count += 1
        pred = model(data.x_dict, data.edge_index_dict)
        test_loss += float(criterion(data.y_dict, pred))  
    test_loss/=count

    print('Epoch:{}, Train loss:{:.6f}, valid loss:{:.6f}'.format(0, float(train_loss), float(test_loss)))
    train_list.append(float(train_loss))
    test_list.append(float(test_loss))  

    for epoch in range(num_epochs):


       # if mode=='train' and epoch==num_epochs-10: optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
        
        train_loss, count = 0, 0
        for data in train_loader:   
            count += 1
            pred = model(data.x_dict, data.edge_index_dict)
         
            loss = criterion(data.y_dict, pred)
             
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += float(loss)

        train_loss/=count
        
        
        test_loss, count = 0, 0
        for data in test_loader:  
        
            count += 1
            pred = model(data.x_dict, data.edge_index_dict)
            test_loss += float(criterion(data.y_dict, pred)) 
            
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
    parser.add_argument("--model_dir", type=str, default='./fecr_model/')
    parser.add_argument("--data_dir", type=str, default='./data/')
    parser.add_argument("--test_dir", type=str, default='./test/')
    
    parser.add_argument("--plot_flag", type=bool, default=False)
    parser.add_argument("--noPDE", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--train_ratio", type=float, default=1)
    args = parser.parse_args()
    
    
    mode = args.mode
    all_id = args.all_id -1
    device = args.device
    
    if mode == 'test': args.model_exist = True
    
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print('torch seed', seed)

    

    
    print('==========  GrainNN specification  =========')
    print('3D grain microstructure evolution')
    print('the mode is: ', mode, ', the model id is: ', all_id)
    print('device: ', args.device)
    print('model already exists, no training required: ', args.model_exist)
    print('no PDE solver required, input is random: ', args.noPDE)
    print('plot GrainNN verus PDE pointwise error: ', args.plot_flag)
    print('\n')
    

    print('************ setup data ***********')
    num_train, num_valid, num_test = 0, 0, 0 
    
    datasets = sorted(glob.glob(args.data_dir + 'case*'))
    
    data_list = []
    
    for case in range(len(datasets)):
        with open(args.data_dir + 'case' + str(case+1) + '.pkl', 'rb') as inp:  
            try:
                data_list = data_list + dill.load(inp)
            except:
                raise EOFError

    
    datasets = sorted(glob.glob(args.test_dir + 'case*'))
    
    test_list = []
    
    for case in range(len(datasets)):
        with open(args.test_dir + 'case' + str(case+1) + '.pkl', 'rb') as inp:  
            try:
                test_list = test_list + dill.load(inp)
            except:
                raise EOFError
                
    
   # random.shuffle(data_list)
    num_train = int(args.train_ratio*len(data_list))
    num_valid = len(data_list) - num_train
    num_test = len(test_list)
   # train_list = data_list[:num_train]
   # test_list = data_list[num_train:]                 
    
    hp = hyperparam(mode, all_id)
    hp.features = data_list[0].features
    hp.targets = data_list[0].targets

    dataset = DynamicHeteroGraphTemporalSignal(data_list)
    heteroData = dataset[0]
    hp.metadata = heteroData.metadata()
    
    print('==========  data information  =========')
    print('number of train, validation, test runs', num_train, num_valid, num_test)
    print('data frames: ', hp.all_frames, '; GrainNN frames: ', hp.frames, \
          '; ratio: ', int((hp.all_frames-1)/(hp.frames-1)))
    print('features: ', [(k, v) for k, v in hp.features.items()])
    print('targets: ', [(k, v) for k, v in hp.targets.items()])
    print('heteroData metadata', heteroData.metadata())
    print('nodes in samples', [(k, v.shape[0]) for k, v in data_list[0].feature_dicts.items()])
    print('\n')
    
    
    
    print('************ setup model ***********')
    print('==========  architecture  ========')
    print('type -- multilayer heterogeous GCLSTM')
    
    print('input window', hp.window,'; output window', hp.out_win)
    print('epochs: ', hp.epoch, '; learning rate: ', hp.lr)
    print('input feature dimension: ', hp.feature_dim)
    print('hidden dim (layer size): ', hp.layer_size, \
          '; number of layers (for both encoder and decoder): ', hp.layers)
    
    
    
    

    model = GrainNN2(hp)
   # print(model)
   # for model_id, (name, param) in enumerate(model.named_parameters()):
   #            print(name, model_id)
   # if mode=='train' or mode == 'test': model = ConvLSTM_seq(hp, device)
   # if mode=='ini': model = ConvLSTM_start(hp, device)
    
   # model = model.double()
    if device=='cuda':
        model.cuda()
        
    
   # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
   # print('total number of trained parameters ', pytorch_total_params)
   # print('\n')
    

    if args.model_exist==False: 
        ## train the model
        train_loader = dataset #DataLoader(dataset, batch_size=1, shuffle=True)
        test_loader = dataset #DataLoader(dataset, batch_size=1, shuffle=True)
        train_list=[]
        test_list=[]
        
        start = time.time()
        
        model = train(model, hp.epoch, train_loader, test_loader)
        x = train.x
        y = train.y
        edge = train.edge
        end = time.time()
        print('training time', end - start)
        
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

        sio.savemat('loss_curve_mode'+mode+'.mat',{'train':train_list,'test':test_list})



'''
        
        
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



'''
    
