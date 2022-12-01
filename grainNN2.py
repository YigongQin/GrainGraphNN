#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:35:27 2022

@author: yigongqin
"""
import argparse, time, glob, dill, random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from data_loader import DynamicHeteroGraphTemporalSignal
from models import GrainNN2
from parameters import regressor, classifier
from graph_datastruct import graph_trajectory
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel


def criterion(data, pred, mask):
   # print(torch.log(pred['edge_event']))
   # print(data['edge_event'])
    

   # print(p)
    if args.loss == 'regression':
   
        return 1000*torch.mean(mask['joint']*(data['joint'] - pred['joint'])**2)

    if args.loss == 'classification':
        z = pred['edge_event']
        y = data['edge_event']
        
        qualified_y = torch.where(y>-1)
        y = y[qualified_y]
        z = z[qualified_y]
        
        classifier = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hp.weight))
        
        return classifier(z, y.type(torch.FloatTensor))
        # return torch.mean(-weight_ratio*y*torch.log(p+1e-10) - (1-y)*torch.log(1+1e-10-p))
        # 1000*torch.mean((data['joint'] - pred['joint'])**2) \
        # + torch.mean((data['grain'] - pred['grain'])**2)

def class_acc(prob, label):
    
    # use PRAUC
    
    prob = torch.sigmoid(torch.cat(prob))
    y = torch.cat(label)

    AUC = 0
    intervals = 10
    P_list, R_list = [], []
    left_bound = 0
 #   for threshold in np.arange(1-interval, -interval, -interval): 
    for i in range(intervals+1): 
        # the first one is all positive, no negative, recall is one
        threshold = 1 - i/intervals

       # print(threshold)
        p = ((prob>threshold)*1).long()
        
       # Positive = sum(y==1)
        TruePositive = sum( (p==1) & (y==1) )
        FalsePositive = sum( (p==1) & (y==0) )
        FalseNegative = sum( (p==0) & (y==1) )
        Precision = TruePositive/(TruePositive + FalsePositive) if TruePositive else 0
        Recall = TruePositive/(TruePositive + FalseNegative) if TruePositive else 0
      #  F1 = 2*Presicion*Recall/(Presicion + Recall) if Presicion + Recall else 0
        
        P_list.append(Precision)
        R_list.append(Recall)
        
        AUC += (Recall-left_bound)*Precision
        left_bound = Recall
       # print(Precision, Recall, left_bound)
    P_list.reverse()    
    R_list.reverse()
  # print(Positive, TruePositive, FalsePositive)
  #  print(Presicion, Recall, F1)
   # return F1 if Positive else -1
    return AUC, P_list, R_list

def unorder_edge(a):
    return set(map(tuple, a))
         
def edge_error_metric(data_edge_index, pred_edge_index):

    E_pp = unorder_edge(data_edge_index['joint', 'connect', 'joint'].detach().numpy().T)
    E_pq = unorder_edge(data_edge_index['joint', 'pull', 'grain'].detach().numpy().T)
    
    E_t_pp = unorder_edge(pred_edge_index['joint', 'connect', 'joint'].detach().numpy().T)
    E_t_pq = unorder_edge(pred_edge_index['joint', 'pull', 'grain'].detach().numpy().T) 

    return 1-len(E_pp.intersection(E_t_pp))/len(E_pp), \
           1-len(E_pq.intersection(E_t_pq))/len(E_pq)


def train(model, num_epochs, train_loader, test_loader):

    model.train()

    optimizer = torch.optim.Adam(model.parameters(),lr=hp.lr) 
                                 #weight_decay=1e-5) # <--

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=hp.decay_step, gamma=0.5, last_epoch=-1)
  #  torch.autograd.set_detect_anomaly(True)

    train_loss, count = 0, 0

    for data in train_loader:  
        train.x = data.x_dict
        train.edge = data.edge_index_dict
        train.y = data.y_dict
        count += 1 #data.batch
        data.to(device)
        pred = model(data.x_dict, data.edge_index_dict)
        train_loss += float(criterion(data.y_dict, pred, data['mask'])) 
    train_loss/=count

    test_loss, count = 0, 0
    for data in test_loader:      
        count += 1
        data.to(device)
        pred = model(data.x_dict, data.edge_index_dict)
        test_loss += float(criterion(data.y_dict, pred, data['mask']))  
    test_loss/=count

    print('Epoch:{}, Train loss:{:.6f}, valid loss:{:.6f}'.format(0, float(train_loss), float(test_loss)))
    train_loss_list.append(float(train_loss))
    test_loss_list.append(float(test_loss))  

    for epoch in range(num_epochs):


       # if mode=='train' and epoch==num_epochs-10: optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
        
        train_loss, count = 0, 0
        train_prob, train_label = [], []
        for data in train_loader:   
            count += 1
            data.to(device)
            pred = model(data.x_dict, data.edge_index_dict)
         
            loss = criterion(data.y_dict, pred, data['mask'])
            
            train_prob.append(pred['edge_event'])
            train_label.append(data.y_dict['edge_event'])

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += float(loss)

        train_loss/=count
        
        
        test_loss, count = 0, 0
        test_prob, test_label = [], []
        for data in test_loader:  
        
            count += 1
            data.to(device)
            pred = model(data.x_dict, data.edge_index_dict)
            test_loss += float(criterion(data.y_dict, pred, data['mask'])) 
                   
            test_prob.append(pred['edge_event'])
            test_label.append(data.y_dict['edge_event'])
            
        test_loss/=count
        

        print('Epoch:{}, Train loss:{:.6f}, valid loss:{:.6f}'.format(epoch+1, float(train_loss), float(test_loss)))
        if args.loss == 'classification':
            train_auc, P_list, R_list = class_acc(train_prob, train_label)
            test_auc, P_list, R_list = class_acc(test_prob, test_label)
            print('Train AUC:{:.6f}, valid AUC:{:.6f}'.format(train_auc, test_auc)) 
        train_loss_list.append(float(train_loss))
        test_loss_list.append(float(test_loss))       
        scheduler.step()
        
    print('model id:', args.model_id, 'loss', test_loss)
    if args.loss == 'classification':
        print('model id:', args.model_id, 'PR AUC', float(test_auc))
        train.plist = P_list
        train.rlist = R_list
    
    return model 




if __name__=='__main__':
    

    
    parser = argparse.ArgumentParser("Train the model.")
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--model_id", type=int, default=0)
    parser.add_argument("--model_exist", type=bool, default=False)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--model_dir", type=str, default='./fecr_model/')
    parser.add_argument("--data_dir", type=str, default='./data/')
    parser.add_argument("--test_dir", type=str, default='./test/')
    parser.add_argument("--model_name", type=str, default='HGCLSTM')
    
    parser.add_argument("--plot_flag", type=bool, default=False)
    parser.add_argument("--noPDE", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--loss", type=str, default='regression')
    args = parser.parse_args()
    
    
    mode = args.mode
    model_id = args.model_id
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
    print('the mode is: ', mode, ', the model id is: ', model_id)
    print('device: ', args.device)
    print('model already exists, no training required: ', args.model_exist)
    print('no PDE solver required, input is random: ', args.noPDE)
    print('plot GrainNN verus PDE pointwise error: ', args.plot_flag)
    print('\n')
    

    print('************ setup data ***********')
    num_train, num_valid, num_test = 0, 0, 0 
    
    
    if mode == 'train':
    
        datasets = sorted(glob.glob(args.data_dir + 'case*'))
        
        data_list = []
        
        for case in datasets:
            with open(case, 'rb') as inp:  
                try:
                    data_list = data_list + dill.load(inp)
                except:
                    raise EOFError

        num_train = int(args.train_ratio*len(data_list))
        num_valid = len(data_list) - num_train
        sample = data_list[0]
        
    if mode == 'test':
        
        datasets = sorted(glob.glob(args.test_dir + 'case*'))
        
        data_list = []
        
        for case in datasets:
            with open(case, 'rb') as inp:  
                try:
                    data_list = data_list + dill.load(inp)
                except:
                    raise EOFError
                
        num_test = len(data_list)
        sample = data_list[0]
        
   # random.shuffle(data_list)

    
    train_list = data_list[:num_train]
    test_list = data_list[num_train:]                 
    
    if args.loss == 'regression':
        hp = regressor(mode, model_id)
    elif args.loss == 'classification':
        hp = classifier(mode, model_id)
    else:
        raise KeyError
        
    hp.features = sample.features
    hp.targets = sample.targets
    hp.device = device

    train_tensor = DynamicHeteroGraphTemporalSignal(train_list)
    test_tensor = DynamicHeteroGraphTemporalSignal(test_list)
    
    if len(test_list)>0:
        heteroData = test_tensor[0]
    else:
        heteroData = train_tensor[0]
    hp.metadata = heteroData.metadata()
    
    print('==========  data information  =========')
    print('data dir', args.data_dir)
    print('test dir', args.test_dir)
    print('number of train, validation, test runs', num_train, num_valid, num_test)
    print('GrainNN frames: ', hp.frames)
    print('features: ', [(k, v) for k, v in hp.features.items()])
    print('targets: ', [(k, v) for k, v in hp.targets.items()])
    print('heteroData metadata', heteroData.metadata())
    print('nodes in samples', [(k, v.shape[0]) for k, v in sample.feature_dicts.items()])
    print('edges in samples', [(k, v.shape[1]) for k, v in sample.edge_index_dicts.items()])
    print('\n')
    
    
    
    print('************ setup model ***********')
    print('==========  architecture  ========')
    print('type -- multilayer heterogeous GCLSTM')
    
    print('input window', hp.window,'; output window', hp.out_win)
    
    print('input feature dimension: ', [(k, len(v)) for k, v in hp.features.items()])
    print('hidden dim (layer size): ', hp.layer_size, \
          '; number of layers (for both encoder and decoder): ', hp.layers)
    print('\n')
    
    
    if mode == 'train':
        print('************ training specification ***********')            
        print('epochs: ', hp.epoch, '; learning rate: ', hp.lr)
        print('batch size: ', hp.batch_size)
        print('loss type: ', args.loss)
        if args.loss == 'classification':
            print('weight of positive event: ', hp.weight)
        print('schedueler step: ', hp.decay_step)
        
        print('\n')
    
    
    

    model = GrainNN2(hp)
   # print(model)
   # for model_id, (name, param) in enumerate(model.named_parameters()):
   #            print(name, model_id)

    
   # model = model.double()
    if device=='cuda':
        model.cuda()
        print('use %d GPUs'%torch.cuda.device_count())
        
   # model = DataParallel(model)
        
    
   # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
   # print('total number of trained parameters ', pytorch_total_params)
   # print('\n')
    

    if args.mode == 'train': 
        ## train the model
        train_loader = DataLoader(train_tensor, batch_size=hp.batch_size, shuffle=True)
        test_loader = DataLoader(test_tensor, batch_size=64, shuffle=False)
        train_loss_list=[]
        test_loss_list=[]
        
        start = time.time()
        
        model = train(model, hp.epoch, train_loader, test_loader)
        x = train.x
        y = train.y
        edge = train.edge
        end = time.time()
        print('training time', end - start)
        
        if mode == 'train': torch.save(model.state_dict(), args.model_dir + args.model_name + str(model_id))
      #  if mode == 'ini': torch.save(model.state_dict(), './ini_lstmmodel'+str(all_id))
        
        fig, ax = plt.subplots() 
        ax.semilogy(train_loss_list)
        ax.semilogy(test_loss_list)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['training loss','validation loss'])
        plt.title('training time:'+str( "%d"%int( (end-start)/60 ) )+'min')
        plt.savefig('loss.png')
        
        if args.loss == 'classification':
        
            fig, ax = plt.subplots() 
            ax.scatter(train.rlist, train.plist)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Plot')
            plt.savefig('PR.png')        
        
            optim_arg = max(range(len(train.plist)), key=lambda i: train.rlist[i]+train.plist[i])
            optim_threshold, optim_p, optim_r = optim_arg/len(train.plist), train.plist[optim_arg], train.rlist[optim_arg]
            print('the optimal threshold for classification is: ', optim_threshold, ', with precision/recall', float(optim_p), float(optim_r))

        with open('loss.txt', 'w') as f:
            f.write('epoch, training loss, validation loss\n' )
            for i in range(len(train_loss_list)):
                f.write("%d  %f  %f\n"%(i, train_loss_list[i], test_loss_list[i]))


    if args.mode == 'test':
        
        model.load_state_dict(torch.load(args.model_dir + args.model_name + str(model_id)))
        model.eval() 
              
        
        for case, data in enumerate(test_tensor):
            print('case %d'%case)
         #   print(pred['joint'])
          #  traj = graph_trajectory(seed = data.physical_params['seed'], frames = 5)
           # traj.load_trajectory(rawdat_dir = '.')
            with open('./edge_data/traj1.pkl', 'rb') as inp:  
                try:
                    traj = dill.load(inp)
                except:
                    raise EOFError

            
            pred = model(data.x_dict, data.edge_index_dict)
            data.x_dict, data.edge_index_dict = model.update(data.x_dict, data.edge_index_dict, pred)
            pp_err, pq_err = edge_error_metric(data.edge_index_dict, data['nxt'])
            
            
            traj.GNN_update( (data.x_dict['joint'][:,:2]).detach().numpy())
           # traj.show_data_struct()
            
            
            print('connectivity error of the graph: pp edge %f, pq edge %f'%(pp_err, pq_err))
          #  print('case %d the error %f at sampled height %d'%(case, traj.error_layer, 0))
            

    
