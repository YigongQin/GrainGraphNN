#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:54:13 2023

@author: yigongqin
"""

from graph_trajectory import graph_trajectory, check_connectivity, use_quadruple_find_joint
from graph_datastruct import newcmp
import matplotlib.pyplot as plt

import h5py, glob, re, argparse, os, dill
import numpy as np
from scipy.interpolate import griddata
from collections import defaultdict
from math import pi
from TemperatureProfile3DAnalytic import ThermalProfile
from user_generate import user_defined_config
from copy import deepcopy




class graph_trajectory_geometric(graph_trajectory):
    def __init__(self, 
                 lxd: float = 40, 
                 randInit: bool = False,
                 seed: int = 1, 
                 noise: float = 0.01, 
                 frames: int = 121,
                 BC: str = 'periodic',
                 adjust_grain_size = False,
                 adjust_grain_orien = False,
                 user_defined_config = None):   
        super().__init__(lxd = lxd, randInit = randInit, seed = seed, noise = noise, frames = frames, BC = BC,\
                         adjust_grain_size  = adjust_grain_size , 
                         adjust_grain_orien = adjust_grain_orien,
                         user_defined_config = user_defined_config)
            

        if user_defined_config:
            self.meltpool = user_defined_config['meltpool']
            self.geometry = user_defined_config['geometry']
        else:   
            self.meltpool = 'cylinder'
            self.geometry = {'z0':6, 'r0':30}
        
        if 'V' in self.physical_params:
            self.geometry['melt_pool_angle'] = np.arcsin(self.physical_params['R']/self.physical_params['V'])
        
        self.PF2Grain = defaultdict(int)
        self.Grain2PF = defaultdict(int)
        self.manifold_normal = {'x':[0], 'z':[0]} # assume no-flux boundary condition, the first grain is always boundary
        self.max_y = self.lyd/self.lxd
    
        
    
    
    def load_pde_data(self, rawdat_dir: str = './'):
       
        
        self.data_file = (glob.glob(rawdat_dir + '/*seed'+str(self.seed)+'_*.h5'))[0]
        f = h5py.File(self.data_file, 'r')
        self.x = np.asarray(f['x_coordinates'])[1:-1]
        self.y = np.asarray(f['y_coordinates'])[1:-1]
        self.z = np.asarray(f['z_coordinates'])[1:-1] 
        self.lyd, self.lzd = self.y[-1], self.z[-1]
        G = float(re.search('G(\d+\.\d+)', self.data_file).group(1))
        R = float(re.search('Rmax(\d+\.\d+)', self.data_file).group(1))
        U = G*R
        
        self.therm = ThermalProfile([self.lxd, self.lyd, self.lzd], [G, R, U], self.seed)
        self.physical_params = {'G':G, 'R':R, 'U':U}
        
        
        assert len(self.x) == self.imagesize[0]
        assert int(self.lxd) == int(round(self.x[-1]))

        self.nx, self.ny = len(self.x), len(self.y)



        angles = np.asarray(f['angles'])
        num_theta = (len(angles)-1)//2
        
        volume = np.asarray(f['volume'])        
        grainToPF = np.asarray(f['grainToPF'])
        
        print('total number of choices of orientations: ', num_theta)
        print('total number of phase fields: ', len(grainToPF))
        
        PF_on_manifold = []
        
        for grain, pf in enumerate(grainToPF):
            self.Grain2PF[grain + 2] = pf
            self.PF2Grain[pf] = grain + 2
    
        self.manifold = np.asarray(f['manifold'])
        
        if self.meltpool == 'cylinder' or self.meltpool == 'moving':
            self.geodesic_y = np.asarray(f['geodesic_y_coors'])
            
            
            self.ny_geod = len(self.geodesic_y)            
           # print(len(self.manifold),4*nx*ny_geod  )
            assert len(self.manifold) == 4*self.nx*self.ny_geod        
            self.manifold = self.manifold.reshape((4, self.ny_geod, self.nx),order='F')       
            self.alpha_pde = self.manifold[-1,:,:].T
            geodesic_length = np.max(self.geodesic_y) - np.min(self.geodesic_y)
          #  self.lyd = geodesic_length
            N = int(np.round(geodesic_length/self.mesh_size))+1
            
            xs0, ys0 = np.meshgrid(self.x, self.geodesic_y, indexing='ij')
            xs, ys = np.meshgrid(self.x, np.linspace(np.min(self.geodesic_y), np.max(self.geodesic_y), N, endpoint=True), indexing='ij')
            points = np.stack([xs0.flatten(), ys0.flatten()]).T
    
            self.euclidean_alpha_pde = griddata(points, self.alpha_pde.flatten(), (xs, ys), method = 'nearest')
           # num_grains_manifold = len(np.unique(self.euclidean_alpha_pde))
           # print('num grains, average size (um): ', num_grains_manifold, np.sqrt(4*geodesic_length*self.lxd**2/num_grains_manifold/pi))
            self.max_y = self.euclidean_alpha_pde.shape[1]/self.euclidean_alpha_pde.shape[0]

                  
            
            
            
                

            
        pfs = np.unique(self.alpha_pde)
        
        self.num_regions = 1
        for pf in grainToPF:
            if pf in pfs:
                
                self.alpha_pde[self.alpha_pde==pf] = self.num_regions + 1
                self.euclidean_alpha_pde[self.euclidean_alpha_pde==pf] = self.num_regions + 1
                PF_on_manifold.append(pf)                
                self.num_regions += 1
        
        
        PF_on_manifold = np.asarray(PF_on_manifold)
        print('number of phase fields on SLI: ', len(PF_on_manifold))
        
        self.theta_x = np.zeros(1 + self.num_regions)
        self.theta_z = np.zeros(1 + self.num_regions)
        
        
        angle_on_manifold = np.zeros_like(PF_on_manifold, dtype=int)
        for i, p in enumerate(PF_on_manifold):
            if p<=num_theta:
                angle_on_manifold[i] = p
            else:
                angle_on_manifold[i] = p%num_theta + 1
        
        self.theta_x[2:] = angles[angle_on_manifold]
        self.theta_z[2:] = angles[angle_on_manifold + num_theta]
        
        
        cur_grain, counts = np.unique(self.euclidean_alpha_pde, return_counts=True)
        self.euclidean_alpha_pde = self.euclidean_alpha_pde.T
        self.area_counts = dict(zip(cur_grain, counts))
        self.layer_grain_distribution()
        
        self.imagesize = (int(np.round(self.lxd/self.mesh_size))+1, int(np.round(geodesic_length/self.mesh_size))+1)
        
    def show_data_struct(self):
       # graph_trajectory.show_data_struct(self)
        print('plot manifold')
        fig, ax = plt.subplots(1, 4, figsize=(20, 5), gridspec_kw={'width_ratios': [1, 1, 1, 1], 'height_ratios': [1]})

        """
        ax[0].plot([0, self.lyd], [0, 0], 'k')
        ax[0].plot([0, self.lyd], [self.lzd, self.lzd], 'k')
        ax[0].plot([0, 0], [0, self.lzd], 'k')
        ax[0].plot([self.lyd, self.lyd], [0, self.lzd], 'k')
        
        y = self.y

        z  = self.geometry['z0'] + self.lzd - np.sqrt(self.geometry['r0']**2 - (y-self.lyd/2)**2)
        y = y[z<self.lzd]
        z = z[z<self.lzd]

        ax[0].plot(y, z, 'k')
        ax[0].axis("equal")
        ax[0].axis('off')  
        
        """


        for coors in self.region_coors.values():
            for i in range(len(coors)):
                cur = coors[i]
                nxt = coors[i+1] if i<len(coors)-1 else coors[0]
                ax[0].plot([cur[0],nxt[0]], [cur[1],nxt[1]], 'k')
                
        x, y = zip(*self.region_center.values())     
    
      #  ax[0].scatter(list(x), list(y), c = 'k')
        ax[0].axis("equal")
        ax[0].axis('off')
       # ax[0].set_title('(Q, V, E)=(%d, %d, %d)'%(Q, V, E))
       # ax[0].set_title('Graph')
        ax[0].set_frame_on(False)
        
        ax[1].imshow(self.theta_z[self.alpha_field]/pi*180, origin='lower', cmap=newcmp, vmin=0, vmax=90)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
      #  ax[1].set_title('GrainGNN') 
        
        ax[2].imshow(self.theta_z[self.euclidean_alpha_pde]/pi*180, origin='lower', cmap=newcmp, vmin=0, vmax=90)
        ax[2].set_xticks([])
        ax[2].set_yticks([])
       # ax[2].set_title('Phase field')         
        
        ax[3].imshow(1*(self.alpha_field!=self.euclidean_alpha_pde),cmap='Reds',origin='lower')
        ax[3].set_xticks([])
        ax[3].set_yticks([])
        p_err = int(np.round(self.error_layer*100))

            
            
        if self.save:
            plt.savefig(self.save, dpi=400)
            
    def createGraph(self, image):
        
        cur_joint = defaultdict(list)
        s = image.shape[0]
        quadraples = defaultdict(list)
        
        
        
        for j in range(1, image.shape[1] -1):
            for i in range(1, image.shape[0] -1):
                occur = {}
                for dj in [-1, 0, 1]:
                    for di in [-1, 0, 1]:
                        if di**2 + dj**2<=1:
                            pixel = image[i+di, j+dj]
                            if pixel in occur:
                                occur[pixel]+=1
                            else:
                                occur.update({pixel:1})
      
                alpha_occur=len(occur)
                max_occur=0
                for k, v in occur.items():
                    max_occur = max(max_occur, v);
        
                if alpha_occur==3 and  max_occur<=5:
                    index = tuple(sorted(list(occur.keys())))
                    if index not in cur_joint or max_occur<cur_joint[index][2]:    
                        cur_joint[index] = [i/s, j/s, max_occur]
                
                if alpha_occur==4: 
                    index = tuple(sorted(list(occur.keys())))
                    if index not in quadraples or max_occur<quadraples[index][2]:    
                        quadraples[index] = [i/s, j/s, max_occur]
        
        if self.BC == 'noflux':
            self.find_boundary_vertex(image, cur_joint) 
        
               
        print('number of quadruples', quadraples)
        

        del_joints = {}
        for q, coors in quadraples.items():
            q_list = list(q)
          #  print(q_list)
            for comb in [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]:
                arg = tuple([q_list[i] for i in comb])

                if arg in cur_joint:

                    del_joints.update({arg:cur_joint[arg]})
                    del cur_joint[arg]
        
       # self.remove_one_two_side_grain(cur_joint)
    
        total_missing, candidates, miss_case  = check_connectivity(cur_joint)
        use_quadruple_find_joint(quadraples, total_missing, cur_joint, miss_case, candidates, del_joints)
        total_missing, candidates, miss_case  = check_connectivity(cur_joint)
        print('total missing edges, ', total_missing, miss_case)
         

        
       # del cur_joint[(371, 381, 385)]
        vert_cnt = 0
        for k, v in cur_joint.items():
            self.vertices[vert_cnt] = v[:2]
            self.vertex2joint[vert_cnt] = set(k)
            vert_cnt += 1
            
        self.joint2vertex = dict((tuple(sorted(v)), k) for k, v in self.vertex2joint.items())
        self.update(init=True)        


    def remove_one_two_side_grain(self, cur_joint):
        remove_list = []
        grain_occur = defaultdict(int)
        for k, v in cur_joint.items():
            for grain in k:
                grain_occur[grain] += 1

        for k, v in grain_occur.items():
            if v<= 2:
                remove_list.append(k)
        replace = []
        for r in remove_list:
            for k, v in cur_joint.copy().items():
                if r in set(k):
                    neigh = set(k)
                    neigh.remove(r)
                    replace.append(list(neigh)[0])
                    break
                
        for i, r in enumerate(remove_list):    
            print(r, replace[i])
            for k, v in cur_joint.copy().items():
                if r in set(k):
                    if replace[i] in set(k):
                        del cur_joint[k]
                        print(k)
                    else:
                        neigh = list(set(k))
                        ind = neigh.index(r)
                        neigh[ind] = replace[i]
                        cur_joint[tuple(sorted(neigh))] = cur_joint[k]
                        print(k, '->', tuple(sorted(neigh)))
                    
        print('remove one two side grain', remove_list)
        
    def cylindrical_y_offset(self):         
        
        if hasattr(self, 'euclidean_alpha_pde'):
            offset = self.geodesic_y[0]
        else:
            offset = - np.arccos( self.geometry['z0']/self.geometry['r0'] )*self.geometry['r0']
                
        return offset
        
    def form_states_tensor(self):

        if self.BC == 'noflux':        
            assert len(self.area_counts) + 1 == self.num_regions
        
        
        self.num_vertices = len(self.joint2vertex)
        
        
        # find normals

        if self.meltpool == 'cylinder' or self.meltpool == 'moving':
            for region in range(1, self.num_regions+1):
                if self.BC == 'noflux' and region == 1:
                    continue
                
                if region in self.region_center:
                    geodesic_y = self.region_center[region][1]*self.lxd + self.cylindrical_y_offset()
                    manifold_normal_z = -geodesic_y/self.geometry['r0']
                else:
                    manifold_normal_z = 0
                self.manifold_normal['x'].append(0)
                self.manifold_normal['z'].append(manifold_normal_z)
        
        super().form_states_tensor(0)
        
if __name__ == '__main__':


    parser = argparse.ArgumentParser("Generate trajectory for irregular grid data")
    parser.add_argument("--mode", type=str, default = 'test')
    parser.add_argument("--rawdat_dir", type=str, default = './cylinder/')
    parser.add_argument("--save_dir", type=str, default = './cylinder/')
    parser.add_argument("--seed", type=int, default = 48)

    parser.add_argument("--boundary", type=str, default = 'noflux')
    parser.add_argument("--size", dest='adjust_grain_size', action='store_true')
    parser.set_defaults(adjust_grain_size=False)    
    parser.add_argument("--orien", dest='adjust_grain_orien', action='store_true')
    parser.set_defaults(adjust_grain_orien=False)

    parser.add_argument("--frame", type=int, default = 121)
    parser.add_argument("--span", type=int, default = 6)
    parser.add_argument("--lxd", type=int, default = 60)
    parser.add_argument("--save_traj", type=bool, default = True)

    
    args = parser.parse_args()

    if args.mode == 'test':                   
        traj = graph_trajectory_geometric(lxd=args.lxd, randInit = False, seed = args.seed, frames = args.frame, BC = args.boundary,
                                          adjust_grain_size = args.adjust_grain_size, 
                                          adjust_grain_orien = args.adjust_grain_orien)
        
        traj.load_pde_data(rawdat_dir = args.rawdat_dir)
        
        
        traj_copy = deepcopy(traj)
       # traj_copy2 = deepcopy(traj)
        
      #  traj_copy2.createGraph(traj_copy2.euclidean_alpha_pde)
        
        traj_copy.createGraph(traj_copy.euclidean_alpha_pde.T)
        
      
        traj.createGraph(traj_copy.alpha_field.T)
       # traj.createGraph(traj.euclidean_alpha_pde)
        
              
        traj.show_data_struct()


    if args.mode == 'generate':   
        
        traj = graph_trajectory_geometric(randInit = True, user_defined_config = user_defined_config())
        traj.show_data_struct()
        
    traj.form_states_tensor()
     
    test_samples = []
    hg0 = traj.states[0]
    
    with open('GR_train_grid.pkl', 'rb') as inp:  
        try:
            GR_grid = dill.load(inp)
        except:
            raise EOFError
    
    G_ = (traj.physical_params['G'] - GR_grid['G_min'])/(GR_grid['G_max'] - GR_grid['G_min'])
    R_ = (traj.physical_params['R'] - GR_grid['R_min'])/(GR_grid['R_max'] - GR_grid['R_min'])
    hg0.span = griddata(np.array([GR_grid['G'], GR_grid['R']]).T, np.array(GR_grid['span']), (G_, R_), method='nearest')
    
    hg0.form_gradient(prev = None, nxt = None, event_list = None, elim_list = None)
    hg0.append_history([])
    test_samples.append(hg0)
    
    G = str(round(traj.physical_params['G'],3))
    R = str(round(traj.physical_params['R'],3))
    
 
    with open(args.save_dir + 'seed' + str(args.seed) + '_G' + G + '_R' + R +\
              '_span' + str(hg0.span) + '.pkl', 'wb') as outp:
        dill.dump(test_samples, outp)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir) 
            
    if args.save_traj:
            with open(args.save_dir + 'traj' + str(args.seed) + '.pkl', 'wb') as outp:
                dill.dump(traj, outp) 


        