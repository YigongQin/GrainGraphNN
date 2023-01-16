#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 21:22:44 2023

@author: yigongqin
"""

import argparse, time, dill, random, os, glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel
from data_loader import DynamicHeteroGraphTemporalSignal
from models import GrainNN_regressor, GrainNN_classifier
from parameters import regressor, classifier_transfered
from graph_trajectory import graph_trajectory
from metrics import feature_metric, edge_error_metric
from QoI import data_analysis


if __name__=='__main__':
    

    parser = argparse.ArgumentParser("Train the model.")

    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--model_dir", type=str, default='./model/')
    parser.add_argument("--truth_dir", type=str, default='./sameGR/level2/')
    parser.add_argument("--regressor_id", type=int, default=41)
    parser.add_argument("--classifier_id", type=int, default=18)
    parser.add_argument("--use_sample", type=str, default='all')
    
    parser.add_argument("--plot_flag", type=bool, default=False)
    parser.add_argument("--noPDE", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=35)


    args = parser.parse_args()
    
    device = args.device
    
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

    print('==========  GrainNN specification  =========')
    print('3D grain microstructure evolution')
    print('device: ', args.device)
    print('no PDE solver required, input is random: ', args.noPDE)
    print('plot GrainNN verus PDE pointwise error: ', args.plot_flag)
    print('torch seed', seed)
    print('\n')
    

    print('************ setup data ***********')

        
    test_list = []
    
    with open('dataset_test.pkl', 'rb') as inp:  
        try:
            test_list = dill.load(inp)
        except:
            raise EOFError
            
    
    sample = test_list[0]

    if args.use_sample !='all':
        
        test_list = test_list[:int(args.use_sample)]
          
    num_test = len(test_list)        

    test_tensor = DynamicHeteroGraphTemporalSignal(test_list)
    heteroData = test_tensor[0]

    print('number of test runs', num_test)

    print('************ setup model ***********')


    
    hp  = regressor(args.regressor_id)
    hpc = classifier_transfered(args.classifier_id)
    

    hp.metadata = heteroData.metadata()    
    hp.features = sample.features
    hp.targets = sample.targets
    hp.device = device

    hpc.metadata = hp.metadata
    hpc.features = hp.features
    hpc.device = hp.device   



    Rmodel = GrainNN_regressor(hp)
    Rmodel.load_state_dict(torch.load(args.model_dir + 'regressor' + str(args.regressor_id)))
    Rmodel.eval() 
    
    
    Cmodel = GrainNN_classifier(hpc, Rmodel)
    Cmodel.load_state_dict(torch.load(args.model_dir + 'classifier' + str(args.classifier_id)))
    Cmodel.eval() 
    
    
    
    
    
    print('==========  architecture  ========')
    print('type -- multilayer heterogeous GCLSTM')

    print('input window', hp.window,'; output window', hp.out_win)
    
    print('input feature dimension: ', [(k, len(v)) for k, v in hp.features.items()])
    print('hidden dim (layer size): ', hp.layer_size, \
          '; number of layers (for both encoder and decoder): ', hp.layers)
        
    print('regressor:')
    print('number of parameters: ')
    print('classifier:')
    print('number of parameters: ')
    print('threshold: ', Cmodel.threshold)
        
    print('\n')
 

    
    print('==========  data information  =========')
    
    print('GrainNN frames: ', hp.frames)
    print('features: ', [(k, v) for k, v in hp.features.items()])
    print('targets: ', [(k, v) for k, v in hp.targets.items()])
    print('heteroData metadata', heteroData.metadata())
    print('nodes in samples', [(k, v.shape[0]) for k, v in sample.feature_dicts.items()])
    print('edges in samples', [(k, v.shape[1]) for k, v in sample.edge_index_dicts.items()])
    print('\n')
    
    

    """
    
    Inference
    
    """      

    traj_list = sorted(glob.glob(args.truth_dir + 'traj*'))
    for case, data in enumerate(test_tensor):
        print('case %d'%case)
     #   print(pred['joint'])
      #  traj = graph_trajectory(seed = data.physical_params['seed'], frames = 5)
       # traj.load_trajectory(rawdat_dir = '.')
       
        with open(traj_list[case], 'rb') as inp:  
            try:
                traj = dill.load(inp)
            except:
                raise EOFError


        for frame in range(hp.frames):
            """
            <1> combine two predictions
            """            

            pred = Rmodel(data.x_dict, data.edge_index_dict)
            pred_c = Cmodel(data.x_dict, data.edge_index_dict)
            pred.update(pred_c)
            
            """
            <2>  update node features
            """
            
            data.x_dict = Rmodel.update(data.x_dict, pred)
            
            """
            <3> predict events and update features and connectivity
            """            
            
            data.x_dict, data.edge_index_dict = Cmodel.update(data.x_dict, data.edge_index_dict, pred)

 
            """
            <4> evaluate
            """
            pp_err, pq_err = edge_error_metric(data.edge_index_dict, data['nxt'])
            
            
            traj.GNN_update( (data.x_dict['joint'][:,:2]).detach().numpy())
           # traj.show_data_struct()
            
            
            print('connectivity error of the graph: pp edge %f, pq edge %f'%(pp_err, pq_err))
          #  print('case %d the error %f at sampled height %d'%(case, traj.error_layer, 0))
            