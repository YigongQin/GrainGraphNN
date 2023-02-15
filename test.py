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
    parser.add_argument("--truth_dir", type=str, default='./debug_set/')
    parser.add_argument("--regressor_id", type=int, default=0)
    parser.add_argument("--classifier_id", type=int, default=1)
    parser.add_argument("--use_sample", type=str, default='all')
    
    parser.add_argument("--plot_flag", type=bool, default=False)

    parser.add_argument('--compare', dest='compare', action='store_true')
    parser.add_argument('--no-compare', dest='compare', action='store_false')
    parser.set_defaults(compare=True)

    args = parser.parse_args()
    
    device = args.device
    

    

    print('==========  GrainNN specification  =========')
    print('3D grain microstructure evolution')
    print('device: ', device)
    print('compare with PDE: ', args.compare)
    print('plot GrainNN verus PDE pointwise error: ', args.plot_flag)
    print('\n')
    

    print('************ setup data ***********')

        
    
    
            
    datasets = sorted(glob.glob(args.truth_dir + 'case*'))

    test_list = []
    for case in datasets:
        with open(case, 'rb') as inp:  
            try:
                test_list = test_list + [dill.load(inp)[0]]
            except:
                raise EOFError
            
    sample = test_list[0]

    if args.use_sample != 'all':
        
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
    Rmodel.load_state_dict(torch.load(args.model_dir + 'regressor' + str(args.regressor_id)+'.pt', map_location=torch.device('cpu')))
    Rmodel.eval() 
    
    
    Cmodel = GrainNN_classifier(hpc, Rmodel)
    Cmodel.load_state_dict(torch.load(args.model_dir + 'classifier' + str(args.classifier_id)+'.pt', map_location=args.device))
    Cmodel.eval() 
    
    Rmodel.threshold = 1e-4 # from P-R plot
    Cmodel.threshold = 0.6  # from P-R plot
    
    if device=='cuda':
        print('use %d GPUs'%torch.cuda.device_count())
        Rmodel.cuda()  
        Cmodel.cuda()
    
    print('==========  architecture  ========')
    print('type -- multilayer heterogeous GCLSTM')

    print('input window', hp.window,'; output window', hp.out_win)
    
    print('input feature dimension: ', [(k, len(v)) for k, v in hp.features.items()])
    print('hidden dim (layer size): ', hp.layer_size, \
          '; number of layers (for both encoder and decoder): ', hp.layers)
        
    print('regressor:')
    print('number of parameters: ')
    print('threshold for grain events: ', Rmodel.threshold)
    print('classifier:')
    print('number of parameters: ')
    print('threshold for edge events: ', Cmodel.threshold)
        
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
    
    test_loader = DataLoader(test_tensor, shuffle=False)
    
    with torch.no_grad():
        for case, data in enumerate(test_loader):
            
            start_time = time.time()
            
            
            data.to(device)
            
            print('case', datasets[case])
            
        
            print('load trajectory for case ', case)
        
            with open(traj_list[case], 'rb') as inp:  
                try:
                    traj = dill.load(inp)
                  #  traj.show_data_struct()
                except:
                    raise EOFError
            
            ''' determine span '''
            
            st_idx = datasets[case].find('span') + 4
            end_idx = datasets[case].find('.')
            span = int(datasets[case][st_idx:-4])
            print('expected span', span)
            
            
            ''' intialization '''
            assert np.all(data['mask']['grain'].detach().numpy()>0)
            
            data['mask']['joint'] = 1 + 0*data['mask']['joint']
           # print(data['mask']['joint'])
            
            grain_event_list = []
            edge_event_list = []       
            
            for frame in range(span, 121, span):
                
                print('================================')
                print('prediction progress %1.2f/1.0'%(frame/120))
               # print(data.x_dict['joint'][:,5])
                """
                <1> combine two predictions
                    a. Rmodel: dx_p, ds & v
                    b. Cmodel: p(i,j), dx for p(i,j)>threshold
                """            


                pred = Rmodel(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
                pred_c = Cmodel(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
                pred.update(pred_c)

                
                """
                <2>  update node features
                     a. x_p, s, v
                     b. z
                """
                
                Rmodel.update(data.x_dict, pred)
                data.x_dict['grain'][:, 2] += span/121
                data.x_dict['joint'][:, 2] += span/121

                
                """
                <3> predict events and update features and connectivity
                    a. E_pp, E_pq
                    b. mask_p, mask_q
                """            
                
                grain_event_truth = set.union(*traj.grain_events[:frame+1])
                grain_event_truth = set([i-1 for i in grain_event_truth])
                
                print(grain_event_truth)
               # print(data['mask']['grain'].shape, pred['grain_area'].shape)
                
                pred['grain_event'] = ((data['mask']['grain'][:,0]>0)&(pred['grain_area']<Rmodel.threshold)).nonzero().view(-1)
                
                pred['grain_event'] = pred['grain_event'][torch.argsort(pred['grain_area'][pred['grain_event']])]
                
               # pred['grain_event'] = torch.tensor([228, 39, 70, 89])
                
                grain_event_list.extend(pred['grain_event'].detach().numpy())
                right_pred_q = len(set(grain_event_list).intersection(grain_event_truth))
                
                print('grain events: ', pred['grain_event'])
                print('grain events hit rate: %d/%d'%(right_pred_q, len(grain_event_truth)) )
                
                
                edge_event_truth = set.union(*traj.edge_events[:frame+1])
               # print(edge_event_truth)

                data.edge_index_dict, pairs = Cmodel.update(data.x_dict, data.edge_index_dict, data.edge_attr_dict, pred, data['mask'])
                pairs = pairs.detach().numpy()
                
                edge_event_list.extend([tuple(i) for i in pairs])
                right_pred_p = len(set(edge_event_list).intersection(edge_event_truth))
                
                print('edge events:', pairs)
                print('edge events hit rate: %d/%d'%(right_pred_p, len(edge_event_truth)//2) )
    

                topogical = len(pred['grain_event'])>0 or len(pairs)>0               
                print('topological changes happens: ', topogical)
                
                print('\n')


                
                """
                <4> evaluate
                """
                
               # print(data['nxt'])
                pp_err, pq_err = edge_error_metric(data.edge_index_dict, data['nxt'])
                print('connectivity error of the graph: pp edge %f, pq edge %f'%(pp_err, pq_err))
                
                X_p = data.x_dict['joint'][:,:2].detach().numpy()

                traj.GNN_update(frame, X_p, data['mask'], topogical, data.edge_index_dict)
                
                
                if True:
                    traj.raise_err = False
                    traj.plot_polygons()
                if True:
                    traj.show_data_struct()
                
                

                """
                <5> next prediction
                    a. x_q
                """
                for grain, coor in traj.region_center.items():
                    data.x_dict['grain'][grain-1, :2] = torch.FloatTensor(coor)
                
                
            
            end_time = time.time()
            print('inference time for case %d'%case, end_time - start_time)            
           
            
            
            
            
            