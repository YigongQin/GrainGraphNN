#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 21:22:44 2023

@author: yigongqin
"""

import argparse, time, dill, random, os, glob
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel
from data_loader import DynamicHeteroGraphTemporalSignal
from models import GrainNN_regressor, GrainNN_classifier
from parameters import regressor, classifier_transfered
from graph_trajectory import graph_trajectory
from metrics import feature_metric, edge_error_metric

from visualization3D.pv_3Dview import grain_visual




def scale_feature_patchs(factor, x_dict, edge_attr_dict):
    
    '''scale edge len'''
    for edge_type in edge_attr_dict:
        edge_attr_dict[edge_type] *= factor
    
    #assert torch.all(x_dict['joint'][:,:2]>=0) and torch.all(x_dict['joint'][:,:2]<=1)
    #assert torch.all(x_dict['grain'][:,:2]>=0) 
    #print('coordinates in bound')
    
    
    x_dict['grain'][:,:2] *= factor
    x_dict['joint'][:,:2] *= factor
    
    domain_offset = torch.floor(x_dict['joint'][:,:2])
    
    x_dict['grain'][:,:2] = x_dict['grain'][:,:2]%1
    x_dict['joint'][:,:2] = x_dict['joint'][:,:2] - domain_offset
    

    return domain_offset






if __name__=='__main__':
    

    parser = argparse.ArgumentParser("Train the model.")

    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--model_dir", type=str, default='./model/')
    parser.add_argument("--truth_dir", type=str, default='./debug_set/9domain/')
    parser.add_argument("--regressor_id", type=int, default=0)
    parser.add_argument("--classifier_id", type=int, default=1)
    parser.add_argument("--use_sample", type=str, default='all')
    parser.add_argument("--stop_frame", type=int, default='0')
    parser.add_argument("--seed", type=str, default='')
    parser.add_argument("--save_fig", type=int, default=0)
    parser.add_argument("--domain_factor", type=int, default=3)   
    parser.add_argument("--size_factor", type=int, default=1)

    
    parser.add_argument("--plot", dest='plot', action='store_true')
    parser.set_defaults(plot=False)

    parser.add_argument("--plot3D", dest='plot3D', action='store_true')
    parser.set_defaults(plot3D=False)
    
    parser.add_argument('--no-compare', dest='compare', action='store_false')
    parser.set_defaults(compare=True)
    
    parser.add_argument('--no-reconstruct', dest='reconstruct', action='store_false')
    parser.set_defaults(reconstruct=True)

    args = parser.parse_args()
    
    device = args.device
    

    

    print('==========  GrainNN specification  =========')
    print('3D grain microstructure evolution')
    print('device: ', device)
    print('compare with PDE: ', args.compare)
    print('plot GrainNN verus PDE pointwise error: ', args.plot)
    print('\n')
    

    print('************ setup data ***********')

        
    
    
            
    datasets = sorted(glob.glob(args.truth_dir + 'seed' + args.seed + '*.pkl'))

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
    
    train_frames = 120
    
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

    traj_list = sorted(glob.glob(args.truth_dir + 'traj'+ args.seed + '*'))
    
    test_loader = DataLoader(test_tensor, shuffle=False)
    
    with torch.no_grad():
        for case, data in enumerate(test_loader):
            

            print('\n')
            print('================================')
            
            print('seed', datasets[case])
            
        
            with open(traj_list[case], 'rb') as inp:  
                try:
                    traj = dill.load(inp)
                  #  traj.show_data_struct()
                except:
                    raise EOFError
            
            grain_seed = traj.physical_params['seed']
            print('load trajectory for seed ', grain_seed)
            
            ''' determine span '''
            
            st_idx = datasets[case].find('span') + 4
            end_idx = datasets[case].find('.')
            span = int(datasets[case][st_idx:-4])
            print('expected span', span)
            traj.span = span
            traj.raise_err = False
            
            
            
            ''' intialization '''
            assert np.all(data['mask']['grain'][:,0].detach().numpy()>0)

            data['mask']['joint'] = 1 + 0*data['mask']['joint']

           # X_p = data.x_dict['joint'][:,:2].detach().numpy()
            traj.extraV_traj = []
            traj.GNN_update(0, data.x_dict, data['mask'], True, data.edge_index_dict, args.compare)
            if args.plot:
                traj.show_data_struct()
                
            if args.save_fig>0:
                traj.save = 'seed' + str(grain_seed) + '_z' + str(0) + '.png'
                traj.show_data_struct()
            
            if args.stop_frame>0:
                traj.frames = args.stop_frame
            traj.final_height = traj.ini_height + (traj.frames -1)/train_frames*(traj.final_height-traj.ini_height)
            
            grain_event_list = []
            edge_event_list = []  
            grain_acc_list = [(traj.ini_height, 0, 0, 0)]
            traj.plot_polygons()
            alpha_field_list = [traj.alpha_field.T.copy()]
            
            layer_err_list = [(traj.ini_height, traj.error_layer)]
            traj.area_traj = traj.area_traj[:1]
            
            if len(traj.grain_events)==0:
                traj.grain_events = [set()]*traj.frames
            
            start_time = time.time()
            
            if args.domain_factor>1:
                domain_offset = scale_feature_patchs(args.domain_factor, data.x_dict, data.edge_attr_dict)
            
            
            data.to(device)
            
            for frame in range(span, traj.frames, span):
                
                
                print('******* prediction progress %1.2f/1.0 ********'%(frame/(traj.frames - 1)))
                height = traj.ini_height + frame/(traj.frames - 1)*(traj.final_height-traj.ini_height) 
                
                """
                <1> combine predictions from regressor and classifier
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
                data.x_dict['grain'][:, 2] += span/(train_frames + 1)
                data.x_dict['joint'][:, 2] += span/(train_frames + 1)

                ''' for time extrapolation, prevent z exceeding 1'''
                if data.x_dict['grain'][0, 2] > train_frames/(train_frames + 1):
                    data.x_dict['grain'][:, 2] = train_frames/(train_frames + 1)         
                    data.x_dict['joint'][:, 2] = train_frames/(train_frames + 1)

                """
                <3> predict events and update features and connectivity
                    a. E_pp, E_pq
                    b. mask_p, mask_q
                """            
           
                pred['grain_event'] = ((data['mask']['grain'][:,0]>0)&(pred['grain_area']<Rmodel.threshold)).nonzero().view(-1)
                
                pred['grain_event'] = pred['grain_event'][torch.argsort(pred['grain_area'][pred['grain_event']])]
                

                data.edge_index_dict, pairs = Cmodel.update(data.x_dict, data.edge_index_dict, data.edge_attr_dict, pred, data['mask'])
                pairs = pairs.detach().numpy()
                

                grain_event_list.extend(pred['grain_event'].detach().numpy())               
                
                print('predicted grain events: ', sorted(grain_event_list))


                topo = len(pred['grain_event'])>0 or len(pairs)>0               
                print('topological changes happens: ', topo)
                
                
                """
                <4> evaluate
                """
                
               # print(data['nxt'])
               # pp_err, pq_err = edge_error_metric(data.edge_index_dict, data['nxt'])
              #  print('connectivity error of the graph: pp edge %f, pq edge %f'%(pp_err, pq_err))
                X = {k:v.clone() for k, v in data.x_dict.items()}
                if args.domain_factor>1:
                    
                    X['joint'][:,:2] =  (X['joint'][:,:2] + domain_offset)/args.domain_factor
                    
 
                traj.GNN_update(frame, X, data['mask'], topo, data.edge_index_dict, args.compare)
                
             
                grain_event_truth = set.union(*traj.grain_events[:frame+1])
                grain_event_truth = set([i-1 for i in grain_event_truth])

                right_pred_q = len(set(grain_event_list).intersection(grain_event_truth))
                grain_acc_list.append((height, len(grain_event_truth), len(grain_event_list), right_pred_q))
                
                print('true grain events: ', sorted(list(grain_event_truth))) 
                print('grain events hit rate: %d/%d'%(right_pred_q, len(grain_event_truth)) ) 
                
                
                if args.reconstruct:
                    traj.plot_polygons()
                    alpha_field_list.append(traj.alpha_field.T.copy())
                
                if args.compare:
                    ''' quantify the image error '''

                    layer_err_list.append((height, traj.error_layer))
  
                    if args.plot:
                        traj.show_data_struct()
                        
                    if args.save_fig>1 and frame%((traj.frames - 1)//(args.save_fig-1))==0:
                        p_err = sum([i[1] for i in layer_err_list])/len(layer_err_list)
                        p_err = int(np.round(p_err*100))
                        traj.save = 'seed' + str(grain_seed) + '_z' + str(height) + '_err' + str(p_err)+'_elimp'+str(right_pred_q)+'_t' + str(len(grain_event_truth)) + '.png'
                        traj.show_data_struct()


                """
                <5> next prediction
                    a. x_q
                    b. edge, edge_attr
                """
                for grain, coor in traj.region_center.items():
                    data.x_dict['grain'][grain-1, :2] = torch.FloatTensor(coor)
                    if args.domain_factor>1:
                        data.x_dict['grain'][grain-1, :2] = (data.x_dict['grain'][grain-1, :2]*args.domain_factor)%1


                data.edge_attr_dict = {}
                for edge_type, index in data.edge_index_dict.items():
                    
                    src, dst = edge_type[0], edge_type[-1]
                    src_idx, dst_idx = index[0], index[-1]
                    
                    src_x = data.x_dict[src][src_idx,:2]
                    dst_x = data.x_dict[dst][dst_idx,:2]
                    
                    rel_loc = src_x - dst_x
                    rel_loc = -1*(rel_loc>0.5) + 1*(rel_loc<-0.5) + rel_loc
                    rel_loc = torch.sqrt(rel_loc[:,0]**2 + rel_loc[:,1]**2).view(-1,1)
            
                    data.edge_attr_dict[edge_type] = rel_loc
 

                
            end_time = time.time()
            print('inference time for seed %d'%grain_seed, end_time - start_time)            
            
           # traj.frames = frame_all + 1
            
            traj.event_acc(grain_acc_list)
            traj.misorientation([i[0] for i in grain_acc_list])
            
            if args.compare:
                
                traj.qoi(mode='graph', compare=True)
                
                traj.layer_err(layer_err_list)
                np.savetxt('seed' + str(grain_seed) + '.txt', layer_err_list)
                
                Gv = grain_visual(seed=grain_seed, height=traj.final_height, lxd=traj.lxd) 
                if args.plot3D:
                    Gv.graph_recon(traj, rawdat_dir=args.truth_dir, span=span, alpha_field_list=alpha_field_list)
            
            else:
                
                traj.qoi(mode='graph', compare=False)
                
                
                traj.x = np.arange(-traj.mesh_size, traj.lxd+2*traj.mesh_size, traj.mesh_size)
                traj.y = traj.x
                traj.z = np.arange(-traj.mesh_size, traj.final_height+2*traj.mesh_size, traj.mesh_size)
                
                Gv = grain_visual(seed=grain_seed, height=traj.final_height, lxd=traj.lxd) 
                if args.plot3D:
                    Gv.graph_recon(traj, rawdat_dir=args.truth_dir, span=span, alpha_field_list=alpha_field_list)
            
'''
                    
#  edge_event_truth = set.union(*traj.edge_events[:frame+1])

#  edge_event_list.extend([tuple(i) for i in pairs])
#  right_pred_p = len(set(edge_event_list).intersection(edge_event_truth))                    

#  print('edge events:', pairs)
#  print('edge events hit rate: %d/%d'%(right_pred_p, len(edge_event_truth)//2) )                    
                  

    init_z = 2
    final_z = 50
    frame_all = 120
'''            