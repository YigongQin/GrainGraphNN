#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 21:22:44 2023

@author: yigongqin
"""

import argparse, time, dill, random, os, glob
import numpy as np
import h5py
import torch
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel
from data_loader import DynamicHeteroGraphTemporalSignal
from models import GrainNN_regressor, GrainNN_classifier
from parameters import regressor, classifier_transfered
from graph_trajectory import graph_trajectory
from graph_trajectory_geometric import graph_trajectory_geometric
from metrics import feature_metric, edge_error_metric

from visualization3D.pv_3Dview import grain_visual
import meshio
from scipy.interpolate import griddata
from tvtk.api import tvtk, write_data
from math import pi

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


def move_to_boundary(joint_coor, E_qp, boundary_max):
    
    boundary_joint = E_qp[1, (E_qp[0]==0).nonzero().view(-1)] 
    for p in boundary_joint:
        dist = torch.tensor([joint_coor[p, 0], boundary_max[0] - joint_coor[p, 0], joint_coor[p, 1], boundary_max[1] - joint_coor[p, 1]])
        if dist.argmin()==0:
            joint_coor[p, 0] = 0
        elif dist.argmin()==1:
            joint_coor[p, 0] = boundary_max[0]
        elif dist.argmin()==2:
            joint_coor[p, 1] = 0
        elif dist.argmin()==3:
            joint_coor[p, 1] = boundary_max[1]

if __name__=='__main__':
    

    parser = argparse.ArgumentParser("Train the model.")

    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--model_dir", type=str, default='./model/')
    parser.add_argument("--truth_dir", type=str, default='./debug_set/all/')
    parser.add_argument("--regressor_id", type=int, default=0)
    parser.add_argument("--classifier_id", type=int, default=1)
    parser.add_argument("--use_sample", type=str, default='all')
    parser.add_argument("--stop_frame", type=int, default='0')
    parser.add_argument("--interp_frames", type=int, default=0)
    parser.add_argument("--seed", type=str, default='10020')
    parser.add_argument("--save_fig", type=int, default=0)
    parser.add_argument("--reconst_mesh_size", type=float, default=0.08)
    parser.add_argument("--domain_factor", type=int, default=1)   
    parser.add_argument("--boundary", type=str, default='periodic')
    parser.add_argument("--nucleation_density", type=float, default=0.00)
    
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


    seed = int(args.seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print('torch seed', seed)   

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
            

    
            imagesize = (0, 0)
            if args.reconstruct:  
                if hasattr(traj, 'max_y'):
                    imagesize = (int(traj.lxd/args.reconst_mesh_size)+1, int(traj.max_y*traj.lxd/args.reconst_mesh_size)+1)
                elif hasattr(traj, 'lyd'):
                    imagesize = (int(traj.lxd/args.reconst_mesh_size)+1, int(traj.lyd/args.reconst_mesh_size)+1)
                else:                        
                    imagesize = (int(traj.lxd/args.reconst_mesh_size)+1, int(traj.lxd/args.reconst_mesh_size)+1)
            
            
            ''' intialization '''
            #assert np.all(data['mask']['grain'][:,0].detach().numpy()>0)

            data['mask']['joint'] = 1 + 0*data['mask']['joint']

           # X_p = data.x_dict['joint'][:,:2].detach().numpy()
            traj.extraV_traj = []
            X = {k:v.clone() for k, v in data.x_dict.items()}
            prev_X, prev_mask, prev_edge_index_dict = {k:v.clone() for k, v in X.items()}, {k:v.clone() for k, v in data['mask'].items()}, {k:v.clone() for k, v in data.edge_index_dict.items()}
            traj.GNN_update(0, X, data['mask'], True, data.edge_index_dict, args.compare)
            if args.plot:
                traj.show_data_struct()
                
            if args.save_fig>0:
                traj.save = 'seed' + str(grain_seed) + '_z' + str(0) + '.png'
                traj.show_data_struct()
            
            if args.stop_frame>0:
                traj.frames = args.stop_frame
            traj.final_height = traj.ini_height + (traj.frames -1)/train_frames*(traj.final_height-traj.ini_height)
            delta_z = (traj.final_height-traj.ini_height)/(traj.frames - 1)
            
            grain_event_list = []
            edge_event_list = []  
            grain_acc_list = [(traj.ini_height, 0, 0, 0)]
            traj.plot_polygons(imagesize)
            alpha_field_list = [traj.alpha_field.T.copy()]
            
            layer_err_list = [(traj.ini_height, traj.error_layer)]
            traj.area_traj = traj.area_traj[:1]
            
            if len(traj.grain_events)==0:
                traj.grain_events = [set()]*traj.frames
            
            start_time = time.time()
            domain_offset = 0 
            if args.domain_factor>1:
                domain_offset = scale_feature_patchs(args.domain_factor, data.x_dict, data.edge_attr_dict)
            
            
            data.to(device)
            
            for frame in range(span, traj.frames, span):                
                
                print('******* prediction progress %1.2f/1.0 ********'%(frame/(traj.frames - 1)))
                height = traj.ini_height + frame*delta_z
                
                """
                <1> combine predictions from regressor and classifier
                    a. Rmodel: dx_p, ds & v
                    b. Cmodel: p(i,j), dx for p(i,j)>threshold
                """            
                edge_index = data.edge_index_dict.copy()
                edge_feature = data.edge_attr_dict.copy()
                if args.boundary == 'noflux':
                    for edge_type, index in data.edge_index_dict.items():
                        
                        src, dst = edge_type[0], edge_type[-1]
                        if src == 'grain':
        
                            edge_index[edge_type] = index[:,(index[0]>0).nonzero().view(-1)] 
                            edge_feature[edge_type] = data.edge_attr_dict[edge_type][(index[0]>0).nonzero().view(-1)] 
                        if dst == 'grain':
                            edge_index[edge_type] = index[:,(index[1]>0).nonzero().view(-1)] 
                            edge_feature[edge_type] = data.edge_attr_dict[edge_type][(index[1]>0).nonzero().view(-1)] 
                        
                pred = Rmodel(data.x_dict, edge_index, edge_feature )
                pred_c = Cmodel(data.x_dict, edge_index, edge_feature )
                pred.update(pred_c)

                
                """
                <2>  update node features
                     a. x_p, s, v
                     b. z
                """
                if hasattr(traj, 'geometry'):
                    curvature = frame*delta_z/traj.geometry['r0']
                else:
                    curvature = 0
                geometry_scaling = {'joint':torch.tensor([1, 1 - curvature]), 
                                    'area':torch.tensor([1 - curvature]), 'volume':torch.tensor([1])}
                Rmodel.update(data.x_dict, pred, geometry_scaling)
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
                if args.boundary == 'noflux': # the grain 0 here is boundary grain
                    pred['grain_event'] = pred['grain_event'][pred['grain_event']!=0]
                
                nucleation_prob = args.nucleation_density*traj.lxd*traj.lxd*delta_z/data['mask']['joint'].sum()
                print('nucleation probability for each junction: ', nucleation_prob)
                data.x_dict, data.edge_index_dict, pairs = Cmodel.update(data.x_dict, data.edge_index_dict, data.edge_attr_dict, pred, data['mask'], nucleation_prob)
               # pairs = pairs.detach().numpy()
                if data.x_dict['grain'].size(dim=0) > traj.num_regions:
                    add_angles = torch.arccos(data.x_dict['grain'][traj.num_regions:, 5]).detach().numpy()
                    traj.theta_z = np.concatenate([traj.theta_z, add_angles])
                    traj.num_regions = data.x_dict['grain'].size(dim=0)

                grain_event_list.extend(pred['grain_event'].detach().numpy())               
                
                print('predicted grain events: ', sorted(grain_event_list))


                topo = len(pred['grain_event'])>0 or len(pairs)>0               
                print('topological changes happens: ', topo)
                
                
                """
                <4> evaluate
                """

                ''' for no flux boundary condition, prevent nodes exceeding boundary and reset the grain 0'''
                
                if args.boundary == 'noflux':
                    data.x_dict['grain'][0, :2] = 0.5
                    data.x_dict['grain'][0, 3:5] = 0
                    data.x_dict['grain'][0, -1] = 0   
                    data.x_dict['joint'][:,:2] = (data.x_dict['joint'][:,:2] + domain_offset)/args.domain_factor
                    
                    max_y = traj.lyd/traj.lxd
                    if hasattr(traj, 'max_y'):
                        max_y = traj.max_y
                    move_to_boundary(data.x_dict['joint'], data.edge_index_dict['grain', 'push', 'joint'], [1, max_y])
                    data.x_dict['joint'][:,0]= torch.clamp(data.x_dict['joint'][:,0], min=0, max=1)
                    data.x_dict['joint'][:,1]= torch.clamp(data.x_dict['joint'][:,1], min=0, max=max_y)
                    
                    assert torch.all(data.x_dict['joint'][:,:2]>-1e-6) and torch.all(data.x_dict['joint'][:,:2]<1+1e-6)
                    data.x_dict['joint'][:,:2] = data.x_dict['joint'][:,:2]*args.domain_factor - domain_offset
                    
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
                    if args.interp_frames>0 and args.boundary == 'noflux':
                        for interp_frame in range(args.interp_frames):                            
                            cur_coeff = (1 + interp_frame)/(1 + args.interp_frames)
                            pre_coeff = 1 - cur_coeff
                            mean_X = {k:v.clone() for k, v in X.items()}
                            mean_X['joint'][:,:2] = cur_coeff*X['joint'][:,:2] + pre_coeff*prev_X['joint'][:,:2] 
                            
                            if cur_coeff>0.5:
                                move_to_boundary(mean_X['joint'], data.edge_index_dict['grain', 'push', 'joint'], [1, max_y])
                            else:
                                move_to_boundary(mean_X['joint'], prev_edge_index_dict['grain', 'push', 'joint'], [1, max_y])
                            
                            mean_X['joint'][:,0]= torch.clamp(mean_X['joint'][:,0], min=0, max=1)
                            mean_X['joint'][:,1]= torch.clamp(mean_X['joint'][:,1], min=0, max=max_y)
                            
                            if cur_coeff>0.5:
                                
                                traj.GNN_update(frame, mean_X, data['mask'], topo, data.edge_index_dict, args.compare)
                            else:
                                traj.GNN_update(frame, mean_X, prev_mask, topo, prev_edge_index_dict, args.compare)
                            traj.plot_polygons(imagesize)
                            alpha_field_list.append(traj.alpha_field.T.copy())
                    traj.plot_polygons(imagesize)
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
 
                prev_X, prev_mask, prev_edge_index_dict = {k:v.clone() for k, v in X.items()}, {k:v.clone() for k, v in data['mask'].items()}, {k:v.clone() for k, v in data.edge_index_dict.items()}
                
            end_time = time.time()
            print('inference time for seed %d'%grain_seed, end_time - start_time)            
            
           # traj.frames = frame_all + 1
            
            traj.event_acc(grain_acc_list)
            
            if args.reconstruct:
                
                
                if hasattr(traj, 'meltpool') and traj.meltpool == 'cylinder':

                    nx, ny, nt = alpha_field_list[0].shape[0], alpha_field_list[0].shape[1],  len(alpha_field_list)
                    x = np.linspace(0, (nx-1)*args.reconst_mesh_size, num = nx, endpoint=True)
                    y = np.linspace(0, (ny-1)*args.reconst_mesh_size, num = ny, endpoint=True) 
                    z = np.linspace(0, (nt-1)*delta_z*span/(1+args.interp_frames), num = nt, endpoint=True)
                    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
                    cartesian_theta = (yy + traj.cylindrical_y_offset())/traj.geometry['r0']
                   # print(cartesian_z[:,:,0], cartesian_z[:,:,-1])
                    cartesian_z = traj.lzd + traj.geometry['z0'] - (traj.geometry['r0'] - zz)*np.cos(cartesian_theta)
                    cartesian_y = traj.lyd/2 + (traj.geometry['r0'] - zz)*np.sin(cartesian_theta) 
                   # print(cartesian_z[:,:,0], cartesian_z[:,:,-1])
                    xx, yy, zz = xx.flatten(order='F'), cartesian_y.flatten(order='F'), cartesian_z.flatten(order='F')
                    alpha_field_save = np.stack(alpha_field_list, axis=-1)
                    alpha_field_save = alpha_field_save.flatten(order='F')


                    alpha_field_save = alpha_field_save[zz<traj.lzd]
                    xx = xx[zz<traj.lzd]
                    yy = yy[zz<traj.lzd]
                    zz = zz[zz<traj.lzd]

        
                    points = np.stack([xx,yy,zz], axis=0).T
                    cells = [("vertex", [[i] for i in range(len(xx))] )]
                    mesh = meshio.Mesh(points, cells, point_data={"alpha":traj.theta_z[alpha_field_save]/pi*180})
                    mesh.write("unstructured_seed"+str(seed)+".vtk")
    

                    """ interpolate to an image """
                    
                    interp_mesh = 0.1
                    xi = np.linspace(0, traj.lxd, num = int(traj.lxd/interp_mesh)+1, endpoint=True)
                    yi = np.linspace(0, traj.lyd, num = int(traj.lyd/interp_mesh)+1, endpoint=True)
                    zi = np.array([traj.lzd])
                    xxi, yyi, zzi = np.meshgrid(xi, yi, zi, indexing='ij')
                    print(points.shape, )
                    alpha_i = griddata(points, alpha_field_save, (xxi, yyi, zzi), method='nearest')                    
                    
                    
                    """
                    interp_mesh = 0.1
                    xi = np.linspace(0, traj.lxd, num = int(traj.lxd/interp_mesh)+1, endpoint=True)
                    yi = np.linspace(0, traj.lyd, num = int(traj.lyd/interp_mesh)+1, endpoint=True)
                    zi = np.linspace(0, traj.lzd, num = int(traj.lzd/interp_mesh)+1, endpoint=True)  
                    xxi, yyi, zzi = np.meshgrid(xi, yi, zi, indexing='ij')
                    print(points.shape, )
                    alpha_i = griddata(points, alpha_field_save, (xxi, yyi, zzi), method='nearest')


                    dist = np.sqrt( (yyi-traj.lyd/2)**2 + (zzi-traj.lzd-traj.geometry['z0'])**2 ) - traj.geometry['r0']
                    alpha_i = traj.theta_z[alpha_i]/pi*180
                    
                    alpha_i[dist>0] = np.nan
                   
                                        
                    grid = tvtk.ImageData(spacing=(interp_mesh, interp_mesh, interp_mesh), origin=(0, 0, 0), 
                           dimensions=alpha_i.shape)
        
                    grid.point_data.scalars = alpha_i.ravel(order='F')
                    grid.point_data.scalars.name = 'theta_z'
                    rawdat_dir = ''
                    dataname = rawdat_dir + 'seed'+str(seed) + '.vtk'
                   #rawdat_dir + 'seed'+str(self.seed)+'_G'+str('%2.2f'%self.physical_params['G'])\
                   #+'_R'+str('%2.2f'%self.physical_params['R'])+'.vtk'
                    write_data(grid, dataname)
                   
                    """
                
               # alpha_field_save = alpha_field_save.flatten(order='F')
                
                if hasattr(traj, 'geometry'):
                    z0 = traj.geometry['z0']
                    r0 = traj.geometry['r0']
                else:
                    z0 = traj.ini_height
                    r0 = 0
                """
                hf = h5py.File('alpha_layer_seed'+str(seed)+'_nx'+str(alpha_field_list[0].shape[0])+
                               '_ny'+str(alpha_field_list[0].shape[1])+'_nt'+str(len(alpha_field_list))+
                               '_z'+str(z0)+'_r'+str(r0)+'.h5', 'w')
                hf.create_dataset('alpha_layer', data=alpha_field_save)
                hf.create_dataset('dx', data=traj.mesh_size)
                hf.create_dataset('dz', data=traj.span*delta_z)
                hf.close()
                """
            if args.compare:
                
                traj.qoi(mode='graph', compare=True)
                traj.misorientation([i[0] for i in grain_acc_list], compare=True)
                
                traj.layer_err(layer_err_list)
                np.savetxt('seed' + str(grain_seed) + '.txt', layer_err_list)
               # np.savetxt('event' + str(grain_seed) + '.txt', grain_acc_list) 
                Gv = grain_visual(seed=grain_seed, height=traj.final_height, lxd=traj.lxd) 
                if args.plot3D:
                    Gv.graph_recon(traj, rawdat_dir=args.truth_dir, span=span, alpha_field_list=alpha_field_list)
            
            else:
                
                traj.qoi(mode='graph', compare=False)
                traj.misorientation([i[0] for i in grain_acc_list], compare=False)
                
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
