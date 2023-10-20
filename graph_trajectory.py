#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 14:55:03 2022

@author: yigongqin
"""

import numpy as np
import h5py
import glob, re, os, argparse, dill, copy
from collections import defaultdict
from termcolor import colored
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24})
import itertools
from graph_datastruct import graph, GrainHeterograph, periodic_move,linked_edge_by_junction, periodic_dist_
from math import pi
from collections import Counter
from scipy.interpolate import griddata
from scipy import stats
import math

def relative_angle(p1, p2):
    
    p1 = periodic_move(p1, p2)
    
    return np.arctan2(p2[1]-p1[1], p2[0]-p1[0])

def check_connectivity(cur_joint):
   # jj_link = 0
   
    candidates = set()
    miss_case = defaultdict(int)
    total_missing = 0
    for k1 in cur_joint.keys():
        num_link = 0
        for k2 in cur_joint.keys(): 
            if k1!=k2 and len( set(k1).intersection(set(k2)) ) == 2:
                num_link += 1
    #            jj_link += 1
              #  print(jj_link, k1, k2)
        if num_link !=3:

            candidates.update(set(k1))
            miss_case.update({k1:3-num_link})
            total_missing += abs(3-num_link)
         #   print('find missing junction link', k1, 3-num_link)
    return total_missing, candidates, miss_case   

def use_quadruple_find_joint(quadraples, total_missing, cur_joint, miss_case, candidates, del_joints):

    for q, coor in quadraples.items():
     #   if set(q).issubset(candidates):

            possible = list(itertools.combinations(list(q), 3))
            for c in miss_case.keys():
                if c in possible:
                    possible.remove(c)

            miss_case_sum = 0

            for i, j in miss_case.items():
                if j<0:
                    return
                if len(set(i).intersection(set(q)))>=2:
                    miss_case_sum += j
                 #   max_case = max(max_case, j)                 
                
            print('using quadraples',q,' to find missing links', miss_case_sum)        
            max_case = 1 if miss_case_sum<4 else 2
            for ans in list(itertools.combinations(possible, max_case)):
                print('try ans', ans)
                for a in ans:
                    if a in del_joints:
                        cur_joint[a] = del_joints[a]
                    else:
                        cur_joint[a] = coor
                cur, _, case_new = check_connectivity(cur_joint)
                if miss_case_sum>0 and cur == total_missing - miss_case_sum and len(case_new)<=len(miss_case):
                    print('fixed!')
                    total_missing = cur
                    break
                else:
                    for a in ans:
                            del cur_joint[a]
                            
class graph_trajectory(graph):
    def __init__(self, 
                 lxd: float = 40,
                 randInit: bool = True,
                 seed: int = 1, 
                 noise: float = 0.01, 
                 frames: int = 1,
                 BC: str = 'periodic',
                 adjust_grain_size = False,
                 adjust_grain_orien = False,
                 physical_params = {},
                 user_defined_config = None):   
        super().__init__(lxd = lxd, randInit = randInit, seed = seed, noise = noise, BC = BC,\
                         adjust_grain_size  = adjust_grain_size , adjust_grain_orien = adjust_grain_orien,
                         user_defined_config = user_defined_config)
        
        if user_defined_config:
            self.physical_params = user_defined_config['physical_parameters']
        else:
            self.physical_params = physical_params
            
        self.joint2vertex = dict((tuple(sorted(v)), k) for k, v in self.vertex2joint.items())
        self.frames = frames # note that frames include the initial condition
        self.load_frames = self.frames
        self.match_graph = True

        self.edge_events = []
        self.grain_events = []
        
        self.show = False
        self.states = []
        
        self.save_frame = [True]*self.frames
        
        self.area_traj = []
        
    def volume(self, mode):
        
        self.volume_traj = []
        
       # if time == -1:
       #     time = self.frames - 1
        
        s = self.imagesize[0]
        self.deltaH = (self.final_height-self.ini_height)/self.mesh_size/(self.frames-1)
        
        
        if mode == 'truth':
            
            area0 = self.totalV_frames[:,0]/np.sum(self.totalV_frames[:,0])*s**2
            
            underlying_grain_volume = 4/3/np.sqrt(pi)*area0**1.5
            self.volume_traj.append(underlying_grain_volume.copy())
            for time in range(self.span, self.frames, self.span):
            
                height = self.ini_height + time/(self.frames-1)*(self.final_height-self.ini_height)
                
                self.grain_volume = self.totalV_frames[:,time] - self.extraV_frames[:,time]
                scale_surface = np.sum(self.grain_volume)\
                                /s**2/(height/self.mesh_size+1)
               # print(scale_surface)
                self.grain_volume = self.grain_volume/scale_surface
               # print(self.grain_volume)
                self.grain_volume += underlying_grain_volume + self.extraV_frames[:,time] - area0*(self.ini_height/self.mesh_size+1)
              #  self.grain_volume = self.grain_volume*self.mesh_size**3
                
                self.volume_traj.append(self.grain_volume.copy())
            
            return

        if mode == 'layer':
            
            '''  '''
            
            self.grain_volume = self.extraV_frames[:,time].copy() 
          
        
        if mode == 'graph':
            
            self.grain_volume = 0*self.extraV_traj[0]
            self.deltaH = self.deltaH*self.span

           # assert len(self.area_traj) == 1 + time//self.span
        for grain, area in self.area_traj[0].items():
             self.grain_volume[grain-1] += 4/3/np.sqrt(pi)*area**1.5 
             
        self.volume_traj.append(self.grain_volume.copy())     
             
        for layer, area_counts in enumerate(self.area_traj[1:]):

                
            for grain, area in self.area_traj[layer].items():
                self.grain_volume[grain-1] += self.deltaH*area/2
         

            for grain, area in area_counts.items():
                self.grain_volume[grain-1] += self.deltaH*area/2
         
            self.volume_traj.append(self.grain_volume.copy()+ self.extraV_traj[layer+1])        
        
    def qoi(self, mode='layer', compare=False):
        
        self.volume(mode)
        grain_size = np.cbrt(6*self.volume_traj[-1]/pi)*self.mesh_size
        self.d_mu = np.mean(grain_size)
        self.d_std = np.std(grain_size)    
        
        step = 1 if self.num_regions>400 else 2
 
        bins = np.arange(0, 20, step)
        
        dis, bin_edge = np.histogram(grain_size , bins, density=True)
        bin_edge = 0.5*(bin_edge[:-1] + bin_edge[1:])
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        ax.plot(bin_edge, dis*np.diff(bin_edge)[0], 'r--', label='GNN')
        ax.set_xlim(0, 20)
        ax.set_xlabel(r'$d\ (\mu m)$')
        ax.set_ylabel(r'$P$')     
        KS = 0 
        if compare:
            self.volume('truth')
            grain_size_t = np.cbrt(6*self.volume_traj[-1]/pi)*self.mesh_size
            d_mu_t = np.mean(grain_size_t)
            err_d = np.absolute(d_mu_t - self.d_mu)/d_mu_t
            print('average grain size , err', d_mu_t, err_d)
            dis_t, bin_edge =  np.histogram(grain_size_t, bins, density=True)
            bin_edge = 0.5*(bin_edge[:-1] + bin_edge[1:])
            KS = stats.ks_2samp(grain_size, grain_size_t)[0]
            KS = round(KS, 3)
            print('KS stats', KS)
            ax.plot(bin_edge, dis_t*np.diff(bin_edge)[0], 'b', label='PF')
            
           # np.savetxt('dis'+str(self.seed)+'.txt', np.vstack([bin_edge, dis*np.diff(bin_edge)[0], dis_t*np.diff(bin_edge)[0]]).T)

        ax.legend(fontsize=20)  
        
        plt.savefig('seed'+str(self.seed)+'_size_dis' + '_KS' + str(KS) +'.png', dpi=400, bbox_inches='tight')


    def load_pde_data(self, rawdat_dir: str = './'):
       
        
        self.data_file = (glob.glob(rawdat_dir + '/*seed'+str(self.seed)+'_*.h5'))[0]
        f = h5py.File(self.data_file, 'r')
        self.x = np.asarray(f['x_coordinates'])
        self.y = np.asarray(f['y_coordinates'])
        self.z = np.asarray(f['z_coordinates']) 

        assert int(self.lxd) == int(self.x[-2])
        
        self.x /= self.lxd; self.y /= self.lxd; self.z /= self.lxd
        fnx, fny = len(self.x), len(self.y)

        assert len(self.x) -2 == self.imagesize[0]
        assert len(self.y) -2 == self.imagesize[1]  
        


        G = re.search('G(\d+\.\d+)', self.data_file).group(1)
        R = re.search('Rmax(\d+\.\d+)', self.data_file).group(1)
        data_frames = int(re.search('frames(\d+)', self.data_file).group(1))+1
        
        self.physical_params = {'G':float(G), 'R':float(R)}
        self.alpha_pde_frames = np.asarray(f['cross_sec'])
        self.alpha_pde_frames = self.alpha_pde_frames.reshape((fnx, fny, data_frames),order='F')[1:-1,1:-1,:]        
        
        self.extraV_frames = np.asarray(f['extra_area'])
        self.extraV_frames = self.extraV_frames.reshape((self.num_regions, data_frames), order='F')        
        self.totalV_frames = np.asarray(f['total_area'])
        self.totalV_frames = self.totalV_frames.reshape((self.num_regions, data_frames), order='F')   
         
        self.num_vertex_features = 8  ## first 2 are x,y coordinates, next 5 are possible phase
        self.active_args = np.asarray(f['node_region'])
        nodes_data = len(self.active_args)//(self.num_vertex_features*data_frames)
        self.active_args = self.active_args.\
            reshape((self.num_vertex_features, nodes_data, data_frames ), order='F')
        self.active_coors = self.active_args[:2,:,:]
        self.active_max = self.active_args[2,:,:]
        self.active_args = self.active_args[3:,:,:]
        
        
    def load_trajectory(self, rawdat_dir: str = './'):
        self.load_pde_data(rawdat_dir)
        
        prev_joint = {k:[0,0,100] for k, v in self.joint2vertex.items()}
        prev_grain = set(np.arange(self.num_regions)+1)
        
        for frame in range(self.load_frames):
           
            
            print('load frame %d'%frame)

            ''' check grain information'''
            
            self.alpha_pde = self.alpha_pde_frames[:,:,frame].T
            cur_grain, counts = np.unique(self.alpha_pde, return_counts=True)
            self.area_counts = dict(zip(cur_grain, counts))
            self.area_traj.append(self.area_counts)
           # self.area_counts = {i:self.area_counts[i] if i in self.area_counts else 0 for i in range(1, self.num_regions+1)}
            cur_grain = set(cur_grain)
            if self.BC == 'noflux':
                cur_grain.add(1)

            eliminated_grains = prev_grain - cur_grain

            self.grain_events.append(eliminated_grains)
            prev_grain = cur_grain
            
            
            if frame>0 and not self.match_graph: continue
            
            ''' check junction information'''
            
            cur_joint = defaultdict(list)
            quadraples = defaultdict(list)
            for vertex in range(self.active_args.shape[1]): 
                max_neighbor = self.active_max[vertex, frame]
                args = set(self.active_args[:,vertex,frame])
                xp, yp = self.x[self.active_coors[0,vertex,frame]], self.y[self.active_coors[1,vertex,frame]]
                if -1 in args: args.remove(-1)
                if not args: continue
                args = tuple(sorted(args))
                
                if len(args)==4: 
                    if args not in quadraples or max_neighbor<quadraples[args][2]:    
                        quadraples[args] = [xp, yp, max_neighbor]
                   # quadraples.append([list(args),[xp, yp, max_neighbor]])
                    continue
                if len(args)>4:
                    print(colored('find more than qudraples', 'red'))
                
                if args not in cur_joint or max_neighbor<cur_joint[args][2]:    
                    cur_joint[args] = [xp, yp, max_neighbor]
            
            if self.BC == 'noflux':
                self.find_boundary_vertex(self.alpha_pde.T, cur_joint)
            
            """
            deal with quadruples 
            
            """

            
            ## delete undetermined junctions from quadruples, add right ones later
            del_joints = {}
            for q, coors in quadraples.items():
                q_list = list(q)
              #  print(q_list)
                for comb in [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]:
                    arg = tuple([q_list[i] for i in comb])

                    if arg not in prev_joint and arg in cur_joint:

                        del_joints.update({arg:cur_joint[arg]})
                        del cur_joint[arg]
                        
            print('quadruples', len(quadraples))
            

            
            total_missing, candidates, miss_case  = check_connectivity(cur_joint)
            print('total missing edges, ', total_missing)
            use_quadruple_find_joint(quadraples, total_missing, cur_joint, miss_case, candidates, del_joints)
            total_missing, candidates, miss_case  = check_connectivity(cur_joint)
            


            
            if self.BC == 'periodic' and len(cur_joint)<2*len(cur_grain):
                total_missing, candidates, miss_case  = check_connectivity(cur_joint)
                for arg, coor in del_joints.items():
                    cur_joint[arg] = coor
                    total_new, candidates, miss_case  = check_connectivity(cur_joint)
                    if total_missing<=total_new:
                        del cur_joint[arg]
                    else:
                        print('add', arg)
               
            
                
            if self.BC == 'periodic' and len(cur_joint)>2*len(cur_grain):
                total_missing, candidates, miss_case  = check_connectivity(cur_joint)
             #   print(miss_case)
                for key in miss_case.keys():
                    
                    joint = cur_joint[key]
                    del cur_joint[key]
                    total_missing, candidates, miss_case  = check_connectivity(cur_joint)
                    if total_missing:
                        cur_joint[key] = joint
                    else:
                        print('delete', key)
                        break
                    
            print('case missed', miss_case)
            print('number of grains in pixels %d'%len(cur_grain))
            print('number of junctions %d'%len(cur_joint))
            assert len(cur_grain)>0, self.seed

            if self.BC == 'periodic' and (len(cur_joint)!=2*len(cur_grain) or len(miss_case)>0):
                print(colored('junction find failed', 'red'))
               # print(len(cur_joint), len(cur_grain))
              #  self.grain_events.append(set())
                self.edge_events.append(set()) 
                self.save_frame[frame] = False
                self.form_states_tensor(frame)
                print('----killed-----')
                
               # exit()
                continue
            

            prev_joint = cur_joint            
           
  
            self.vertex_matching(frame, cur_joint, eliminated_grains)
        
            self.update()
            self.form_states_tensor(frame)
          #  if self.error_layer>0.08:
          #      self.save_frame[frame] = False

           # if frame%24==0:    
            #if self.show == True:
           #     self.save = 'data_seed'+str(self.seed)+'_frame'+str(frame)+'.png'
            #    self.show_data_struct()   
                
            print('====================================')  
            print('\n') 
              
                
    def vertex_matching(self, frame, cur_joint, eliminated_grains):
      
      
        print('\n')
        print('summary of event from frame %d to frame %d'%(frame-1, frame))



        def add_edge_event(old_junction_i, old_junction_j):
            vert_old_i = self.joint2vertex[old_junction_i]
            vert_old_j = self.joint2vertex[old_junction_j]                
            switching_edges.add((vert_old_i, vert_old_j))
            switching_edges.add((vert_old_j, vert_old_i))
         #   self.edge_labels[(vert_old_i, vert_old_j)] = 1
         #   self.edge_labels[(vert_old_j, vert_old_i)] = 1  


        def quadruple_(junctions):
            quadraples = {}
            pairs =  set()
            for i in junctions:
                for j in junctions:
                    if len( set(i).difference(set(j)) )==1:
                        if (j, i) not in pairs:
                            pairs.add((i,j))
                            quadraples[tuple(sorted(set(i).union(set(j))))] = (i,j)            
            
            return quadraples


        def perform_switching(old_junction_i, old_junction_j, new_junction_i, new_junction_j):
            
            vert_old_i = self.joint2vertex[old_junction_i]
            vert_old_j = self.joint2vertex[old_junction_j]             
            N_i = [i[0] for i in self.edges if i[1]==vert_old_i]
            N_j = [i[0] for i in self.edges if i[1]==vert_old_j]
            N_i.remove(vert_old_j)
            N_j.remove(vert_old_i)
            N_i, N_j = list(N_i), list(N_j)
            if len(set(self.vertex2joint[N_i[1]]).intersection(set(new_junction_i)))==2:
                N_i.reverse()
            if len(set(self.vertex2joint[N_j[1]]).intersection(set(new_junction_j)))==2:
                N_j.reverse()
            

         #   print(N_i, N_j)
            self.edges[self.edges.index([vert_old_i, N_i[1]])] = [vert_old_i, N_j[1]]
            self.edges[self.edges.index([vert_old_j, N_j[1]])] = [vert_old_j, N_i[1]]
            self.edges[self.edges.index([N_i[1], vert_old_i])] = [N_j[1], vert_old_i]
            self.edges[self.edges.index([N_j[1], vert_old_j])] = [N_i[1], vert_old_j]            
            
            self.joint2vertex[new_junction_i] = self.joint2vertex.pop(old_junction_i)
            self.joint2vertex[new_junction_j] = self.joint2vertex.pop(old_junction_j)
            

            
            
            print( (vert_old_i, vert_old_j), 'neighor switching: ', old_junction_i, old_junction_j, ' --> ', new_junction_i, new_junction_j)
            
            if old_junction_i in old_joint: old_joint.remove(old_junction_i)
            if old_junction_j in old_joint: old_joint.remove(old_junction_j)
            if new_junction_i in new_joint: new_joint.remove(new_junction_i)
            if new_junction_j in new_joint: new_joint.remove(new_junction_j)    


        """
        
        E0: vertex moving
        
        """
        

        
        for k, v in cur_joint.items():
            assert len(v)>=2
            cur_joint[k] = v[:2]

 
        
        old_vertices = self.vertices.copy()
        self.vertices.clear()

        print('Expect %d junctions removed'%(2*len(eliminated_grains)))
        
        
        
        print('\nE0:')
        
        def match():
            old_map = self.joint2vertex.copy()
            new_map = cur_joint.copy()
     
            for joint in self.joint2vertex.keys():
                if joint in cur_joint:        
                    del old_map[joint]
                    del new_map[joint]
                    
            return old_map, new_map
                
        old_map, new_map = match()
        print('number of moving vertices', len(self.joint2vertex) - len(old_map))


        """
        
        E1: neighbor switching
        
        """          
        print('\nE1:')
        
        
        old = set(old_map.keys())
        new = set(new_map.keys())
        
        switching_edges = set() 
        

        
        if old!= new:
            
            
            old_joint = list(old-new)
            new_joint = list(new-old)
           # print('dispearing joints ', len(old_joint), ';  ' , old_joint)
          #  print('emerging joints', len(new_joint ), ';  ' , new_joint)
            
          #  assert len(old_joint) == len(new_joint), "lenght of old %d, new %d"%(len(old_joint), len(new_joint))
            
            quadraples = quadruple_(old_joint)
            quadraples_new = quadruple_(new_joint)
            
            switching_event = set(quadraples.keys()).intersection(set(quadraples_new.keys()))
          #  print(switching_event) 
                                
            for e2 in switching_event:
                
                old_junction_i, old_junction_j = quadraples[e2]
                new_junction_i, new_junction_j = quadraples_new[e2]
      
        
                old_i_x, old_j_x = old_vertices[self.joint2vertex[old_junction_i]], \
                                   old_vertices[self.joint2vertex[old_junction_j]] 
                new_i_x, new_j_x = cur_joint[new_junction_i][:2], cur_joint[new_junction_j][:2]                   
      
                
                
               # print(relative_angle(old_i_x, old_j_x), relative_angle(new_i_x, new_j_x))
                if abs(relative_angle(old_i_x, old_j_x) - relative_angle(new_i_x, new_j_x))>pi/2:
                #    print(colored('switch junction for less rotation', 'green'), new_junction_i, new_junction_j)
                    new_junction_i, new_junction_j = new_junction_j, new_junction_i
                    
                add_edge_event(old_junction_i, old_junction_j)
                perform_switching(old_junction_i, old_junction_j, new_junction_i, new_junction_j)
            
            
            quadraples = quadruple_(old_joint)
            quadraples_new = quadruple_(new_joint)
 


        
        """
        
        E2: grain elimination
        
        """
        
        old_map, new_map = match()
      #  print(old_map, new_map )
        
        if len(eliminated_grains)>0:
            print('\nE2 grain_elimination: ', eliminated_grains)
            
        grain_grain_neigh = {}
        # step 1 merge grains to be eliminated
        for elm_grain in eliminated_grains:
            junction = set()
            for k, v in self.joint2vertex.items():
                if elm_grain in set(k):
                    junction.update(set(k))
            junction.remove(elm_grain) 
            grain_grain_neigh[elm_grain] = junction
      #  print(grain_grain_neigh)
        
        gg_merged = {}
        visited = set()
        for k1, v1 in grain_grain_neigh.items():
            ks, vs = [k1], v1
            for k2, v2 in grain_grain_neigh.items():
                if k1 != k2 and k2 not in visited:
                    if k1 in v2:
                        ks.append(k2)
                        vs.update(v2)
                        
                        visited.add(k2)
            if k1 not in visited:
                gg_merged[tuple(ks)] = vs
            visited.add(k1)            
            
            
     #   print(gg_merged)
      #  left_over = -1
        """
        l = list(self.edges.values())
        lsrc = [i[0] for i in l]
        ldst = [i[1] for i in l]
        e_key = list(self.edges.keys())
        """
        
        for elm_grain, junction in gg_merged.items():
 
            old_vert = []
            todelete = set()
            toadd = []
        #    junction = set()
            for k, v in self.joint2vertex.items():
                
                if len(set(elm_grain).intersection(set(k)))>0:
             #       junction.update(set(k))
                    old_vert.append(v)
                    todelete.add(k)
            
            
            for k, v in new_map.items():
                if set(k).issubset(junction):# and k not in self.joint2vertex:
                    
                    toadd.append(k)                    

            

            
            if len(old_vert) == len(toadd) + 2 :


                visited_joint = {}
                remove_vert = []
    
                for vert in old_vert:
                    N_vert = [i[0] for i in self.edges if i[1]==vert]
                    for neigh in N_vert: #self.vertex_neighbor[vert]:
                        if neigh not in old_vert:
                            for joint in toadd:
                                if len(set(joint).intersection(set(self.vertex2joint[neigh])))==2:
                                    # find new link
                                    if joint in visited_joint:
                                        remove_vert.append([vert, visited_joint[joint]])
                                    else:
                                        visited_joint.update({joint:vert})
                                        break

                ''' remove vertices connect to elim grains '''     
                print(elm_grain,'th grain eliminated with no. of sides %d'%len(todelete), junction)
                for k in todelete:
                    del self.joint2vertex[k]       
                    

                ''' add vertices '''     
                for joint, vert in visited_joint.items():
                    self.joint2vertex[joint] = vert
                    print('the new joint', joint, 'inherit the vert', vert)
                    
            #    print(remove_vert)     
 
                for v1 in old_vert:
                    for v2 in old_vert:
                        if [v1, v2] in self.edges:
                            self.edges[self.edges.index([v1, v2])] = [-1, -1]
                            self.edges[self.edges.index([v2, v1])] = [-1, -1]
                            
                for k1 in visited_joint:
                    for k2 in visited_joint: 
                        if k1 != k2 and len( set(k1).intersection(set(k2)) ) == 2:
                            v1 = visited_joint[k1]
                            v2 = visited_joint[k2]
                            if [v1, v2] not in self.edges:
                                self.edges.append([v1, v2])
                                self.edges.append([v2, v1])     


                def elim_edge(o1, o2, r1, r2):
                    N1 = [i for i, x in enumerate(self.edges) if x[1] == o1 ] 
                    for i in N1:
                        src = self.edges[i][0]

                        if src == o2:
                            self.edges[i] = [-1, -1]
                            
                        elif src in old_vert:
                            idx = self.edges.index([o1, src])    
                            self.edges[i] = [-1, -1]
                            self.edges[idx] = [-1, -1]
                            
                        else:
                            idx = self.edges.index([o1, src]) 
                            
                            self.edges[i] = [src, r1]
                            self.edges[idx] = [r1, src]
                            
                        #    print(o1, src, 'replace by', r1, src)

                try: 
                    o1, o2 = remove_vert[0][0], remove_vert[1][0]
                    r1, r2 = remove_vert[0][1], remove_vert[1][1]
                  #  print(o1, o2, old_vert)  
                    old_vert.remove(o1)
                    old_vert.remove(o2) 
                    elim_edge(o1, o2, r1, r2)
                    elim_edge(o2, o1, r2, r1)
                    
                except:
                    pass
         #   assert len(old_vert) == 2

            
        
        self.edge_events.append(switching_edges)    
        print('number of E2 %d, number of E1 %d'%(len(eliminated_grains), len(switching_edges)//2))

        match = True
        todelete = []
        for joint in self.joint2vertex.keys():
            if joint in cur_joint:
                vert = self.joint2vertex[joint]
                coors = cur_joint[joint]
                if self.BC == 'periodic':
                    self.vertices[vert] = periodic_move(coors, old_vertices[vert])
                else: 
                    self.vertices[vert] = coors
      
            else:
                match = False
                vert = self.joint2vertex[joint]
                #self.vertices[vert] = old_vertices[vert]
                print(colored('disappeared joint detected: ', 'red'), joint, self.joint2vertex[joint])
                ''' cannot resolve, give up the vertex'''
                todelete.append(joint)
        
        
        
        for joint in todelete:
                del self.joint2vertex[joint]
               # del self.vertices[vert]
                
        for joint in cur_joint.keys():
            if joint not in self.joint2vertex:
                match = False
                print(colored('emerged joint detected: ', 'green'), joint)
                self.joint2vertex[joint] = self.num_vertices
                self.vertices[self.num_vertices] = cur_joint[joint]
                self.num_vertices += 1
        
        self.vertex2joint = dict((v, k) for k, v in self.joint2vertex.items())
        ''' ensure the edge connectivity is correct '''
        
       # if not match, fix the edges:

        for k1, v1 in self.joint2vertex.items():
            for k2, v2 in self.joint2vertex.items():
                if k1!=k2 and linked_edge_by_junction(k1, k2):
                    if [v1, v2] not in self.edges:
                        self.edges.append([v1, v2])
                        
        for i, (src, dst) in enumerate(self.edges):
            if src>-1:
                if src in self.vertex2joint and dst in self.vertex2joint:
                    if not linked_edge_by_junction(self.vertex2joint[src], self.vertex2joint[dst]):
                        self.edges[i] = [-1, -1]                        
                else:
                    self.edges[i] = [-1, -1]

                        
    def event_acc(self, events):
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        z_sam = [i[0] for i in events]
        ax.plot(z_sam, [i[1] for i in events], 'b')
        ax.plot(z_sam, [i[2] for i in events], 'r')
        ax.plot(z_sam, [i[3] for i in events], 'r--')
        ax.set_xlabel(r'$z_l\ (\mu m)$')
        ax.set_ylabel('# grain eliminations')
        ax.legend(['PF', 'GNN', 'GNN TP'], fontsize=20)        
        plt.savefig('seed'+str(self.seed)+'_event_acc.png', dpi=400, bbox_inches='tight')

    def layer_err(self, events):
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        z_sam = [i[0] for i in events]
        ax.plot(z_sam, [i[1] for i in events], 'b')
      #  ax.plot(z_sam, [i[2] for i in events], 'r')
        ax.set_xlabel(r'$z_l\ (\mu m)$')
        ax.set_ylabel('MR')
       # ax.legend(['overall', 'event'])        
        plt.savefig('seed'+str(self.seed)+'_layer_err.png', dpi=400, bbox_inches='tight')

    def misorientation(self, z_sam, compare=False):
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        
        misangles = 45 - np.absolute(180/pi*self.theta_z[1:] - 45)
        
        if compare:
            self.volume('truth')
            ax.plot(z_sam, [ np.sum(misangles*i)/np.sum(i) for i in self.volume_traj], 'b', label='PF')
        self.volume('graph')
        mis = [ np.sum(misangles*i)/np.sum(i) for i in self.volume_traj]
        #print(mis)
        ax.plot(z_sam, mis, 'r--', label='GNN')
        ax.set_xlabel(r'$z_l\ (\mu m)$')
        ax.set_ylabel(r'$\Delta \theta$')
        ax.legend(fontsize=20)  
        
        plt.savefig('seed'+str(self.seed)+'_lmisorien.png', dpi=400, bbox_inches='tight')


    def show_events(self):
        

        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot([len(i) for i in self.grain_events])
        ax.plot([len(i)//2 for i in self.edge_events])
        ax.set_xlabel('snapshot')
        ax.set_ylabel('# events')
        ax.legend(['grain elimination', 'neighbor switching'])
        
    def form_states_tensor(self, frame):
        
        hg = GrainHeterograph()
        grain_state = np.zeros((self.num_regions, len(hg.features['grain'])))
        joint_state = np.zeros((self.num_vertices, len(hg.features['joint'])))
        grain_mask = np.zeros((self.num_regions, 1), dtype=int)
        joint_mask = np.zeros((self.num_vertices, 1), dtype=int)
        
        s = int(np.round(self.patch_size/self.mesh_size))+1
        

        
        for grain, coor in self.region_center.items():
            grain_state[grain-1, 0] = coor[0]
            grain_state[grain-1, 1] = coor[1]
            if grain in self.area_counts:
                grain_state[grain-1, 3] = self.area_counts[grain]/s**2
            else:
                grain_state[grain-1, 3] = 0
            grain_mask[grain-1, 0] = 1

        grain_state[:, 2] = frame/self.frames

      #  grain_state[:, 3] = np.array(list(self.area_counts.values())) /s**2
        if frame>0:
            grain_state[:, 4] = self.extraV_frames[:, frame]/s**3
        
        if hasattr(self, 'manifold_normal'):
            self.theta_x[1:] = (self.theta_x[1:] - np.array(self.manifold_normal['x']))%(pi/2) 
            self.theta_z[1:] = (self.theta_z[1:] - np.array(self.manifold_normal['z']))%(pi/2) 
            
            
        grain_state[:, 5] = np.cos(self.theta_x[1:])
        grain_state[:, 6] = np.sin(self.theta_x[1:])
        grain_state[:, 7] = np.cos(self.theta_z[1:])
        grain_state[:, 8] = np.sin(self.theta_z[1:])
        
        # for no flux BC, the first grain is always the boundary grain
        if self.BC == 'noflux':
            grain_state[0, 0:2] = 0.5
            grain_state[0, 3:5] = 0
            grain_state[0, 5:9] = math.sqrt(2)/2
            
        
        
        for joint, coor in self.vertices.items():
            joint_state[joint, 0] = coor[0]
            joint_state[joint, 1] = coor[1]
            joint_mask[joint, 0] = 1
        
        joint_state[:, 2] = frame/self.frames
        joint_state[:, 3] = 1 - self.physical_params['G']/10 #1 - np.log10(self.physical_params['G'])/2
        joint_state[:, 4] = self.physical_params['R']/2
        
        
        gj_edge = []
        gj_len, jj_len = [], []
        for grains, joint in self.joint2vertex.items():
            for grain in grains:
                gj_edge.append([grain-1, joint])
                gj_len.append(periodic_dist_(self.vertices[joint], self.region_center[grain]))
        
        jg_edge = [[joint, grain] for grain, joint in gj_edge]
        jj_edge = [[src, dst] for src, dst in self.edges if src>-1 and dst>-1]
        
        for src, dst in self.edges:
            if src>-1 and dst>-1:
                jj_len.append(periodic_dist_(self.vertices[src], self.vertices[dst]))
            else:
                jj_len.append(-2.0)
      #  jj_len  = [periodic_dist_(self.vertices[src], self.vertices[dst]) for src, dst in self.edges if src>-1 and dst>-1]
        
        hg.feature_dicts.update({'grain':grain_state})
        hg.feature_dicts.update({'joint':joint_state})
        hg.edge_index_dicts.update({hg.edge_type[0]:np.array(gj_edge).T})
        hg.edge_index_dicts.update({hg.edge_type[1]:np.array(jg_edge).T})
        hg.edge_index_dicts.update({hg.edge_type[2]:np.array(jj_edge).T})
        
        
        hg.mask = {'grain':grain_mask, 'joint':joint_mask}
        

        for k, v in self.vertex_neighbor.items():
            if len(v)<3: print(colored('junction with less than three junction neighbor', 'red'), k)
            if len(v)>3: print(colored('junction with more than three junction neighbor', 'red'), k)
            assert len(v)==3
        
        hg.vertex2joint = self.vertex2joint.copy()

        
        hg.physical_params = self.physical_params
        hg.physical_params.update({'seed':self.seed, 'height':frame})

      #  if frame>0:
      #      hg.edge_rotation = np.array(list(self.edge_labels.values()))

      #  assert len(jj_edge) == len(jj_len)
        hg.edge_weight_dicts = {hg.edge_type[0]:np.array(gj_len)[:,np.newaxis],
                                hg.edge_type[1]:np.array(gj_len)[:,np.newaxis],
                                hg.edge_type[2]:np.array(jj_len)[:,np.newaxis]}
        hg.edges = self.edges.copy()
        
        self.states.append(hg) # states at current time

        # reset the edge_labels, check event at next snapshot
        # self.edge_labels = {(src, dst):0 for src, dst in jj_edge} 

    def GNN_update(self, frame, x_dict, mask, topo, edge_index_dict, compare):
        
        
        
        """
        Input:
            updated coordinates of joints, mask (which joint exists), edges
        Output:
            self.vertices, self.vertex2joint(& reverse), self.edges
        """     
 
        X_j = x_dict['joint'][:,:2].detach().numpy()
        X_g = x_dict['grain'][:,3:5].detach().numpy()      

       # print(np.all(X_j<1.5))
       # assert np.all(X_j<1.5) and np.all(X_j>-0.5)
    
        mask_j = mask['joint'][:,0]
        mask_g = mask['grain'][:,0].detach().numpy()
        
        if compare:
            self.alpha_pde = self.alpha_pde_frames[:,:,frame].T
        
        self.vertices.clear()
        for i, coor in enumerate(X_j):
            if mask_j[i] == 1:
                self.vertices[i] = coor

                    
        ''' qoi '''      
        s = (self.patch_size/self.mesh_size)+1
        area_counts = {}
        area_sum = np.sum(X_g[:,0]*mask_g)/(self.lxd/self.patch_size)**2
        for idx, area in enumerate(X_g[:,0]):
            if mask_g[idx]>0:
                area_counts[idx+1] = area*s**2/area_sum
        
        self.extraV_traj.append(mask_g*X_g[:,1]/self.states[0].targets_scaling['grain']*s**3)
        self.area_traj.append(area_counts)
                
       
        
        if topo:
            
            jj_edge = edge_index_dict['joint', 'connect', 'joint'].detach().numpy()
            gj_edge = edge_index_dict['grain', 'push', 'joint'].detach().numpy()
            #print(jj_edge.shape, gj_edge.shape)
            
            self.vertex2joint = defaultdict(set)
            self.edges.clear()
            
            
            for grain, joint in gj_edge.T:
              #  if mask_g[grain]>0 and mask_j[joint]>0:
                    self.vertex2joint[joint].add(grain+1) 

            for k, v in self.vertex2joint.items():
                assert len(v)==3, (k, v)
            """
            print(len(jj_edge.T), len(gj_edge.T), len(self.vertex2joint))
            
            tri = {i:tuple(j) for i,j in self.vertex2joint.items()}    
            
            counts = Counter(tri.values())
             
            # Create a new dictionary with only the keys whose value has a count greater than 1
            result = {k: v for k, v in tri.items() if counts[v] > 1}  
            print(result)
            
            """
            
            
            self.joint2vertex = dict((tuple(sorted(v)), k) for k, v in self.vertex2joint.items())
            self.vertex2joint = dict((v, k) for k, v in self.joint2vertex.items())
           # print(len(jj_edge[0]))
            self.edges = [[i,j] for i, j in jj_edge.T] #[[i,j] for i, j in jj_edge.T if mask_j[i] and mask_j[j]]
            
            
            """
            print(len(self.edges), len(gj_edge.T), len(self.vertex2joint), len(self.joint2vertex))
            edges = [tuple(i) for i in self.edges]
            links = [tuple(i) for i in gj_edge.T]
            print('dup in jj edges', [(item, count) for item, count in Counter(edges).items() if count > 1])
            print('dup in gj edges', [(item, count) for item, count in Counter(links).items() if count > 1])
            """
        
        self.update()


        


if __name__ == '__main__':


    parser = argparse.ArgumentParser("Generate heterograph trajectory")
    parser.add_argument("--mode", type=str, default = 'check')
    parser.add_argument("--rawdat_dir", type=str, default = './')
    parser.add_argument("--save_dir", type=str, default = './')
    parser.add_argument("--seed", type=int, default = 0)
    parser.add_argument("--G", type=float, default = 2)
    parser.add_argument("--R", type=float, default = 0.4)
    parser.add_argument("--boundary", type=str, default = 'periodic')
    parser.add_argument("--size", dest='adjust_grain_size', action='store_true')
    parser.set_defaults(adjust_grain_size=False)    
    parser.add_argument("--orien", dest='adjust_grain_orien', action='store_true')
    parser.set_defaults(adjust_grain_orien=False)


    parser.add_argument("--frame", type=int, default = 121)
    parser.add_argument("--span", type=int, default = 6)
    parser.add_argument("--lxd", type=int, default = 40)
    parser.add_argument("--regenerate", type=bool, default = True)
    parser.add_argument("--save_traj", type=bool, default = True)
    parser.add_argument("--prev", type=int, default = 0)
    args = parser.parse_args()


    """
    this script generates graph trajectory objects and training/testing data 
    the pde simulaion data is in rawdat_dir
    save_dir: save graph data to dir
    seed: realization seed, each graph trajectory relates to one pde simulation

    """
    
    
    
    if args.mode == 'train':
    
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
  
        for seed in [args.seed]:
            
            train_samples = []
            
            if args.regenerate:
              
                traj = graph_trajectory(lxd=args.lxd, seed = seed, frames = args.frame)
              #  traj.update()
              #  traj.show_data_struct()
          
                traj.load_trajectory(rawdat_dir = args.rawdat_dir)
                #traj.show_data_struct()
                if args.save_traj:
                    with open(args.save_dir + 'traj' + str(seed) + '.pkl', 'wb') as outp:
                        dill.dump(traj, outp)
            
            else:
                
                
                with open(args.save_dir + 'traj' + str(seed) + '.pkl', 'rb') as inp:  
                    try:
                        traj = dill.load(inp)
                    except:
                        raise EOFError

            G = str(int(10*traj.physical_params['G']))
            R = str(int(10*traj.physical_params['R']))
            edgeE = str(len(set.union(*traj.edge_events)))
            grainE = str(len(set.union(*traj.grain_events)))
            
            choices = [6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120]
            
            edge_expandstep = 6*360/int(edgeE) if int(edgeE)>0 else 1000
            grain_expandstep = 6*90/int(grainE) if int(grainE)>0 else 1000
            
            for c in choices:
                if c < edge_expandstep and c < grain_expandstep:
                    args.span = c
            
            print('calibrated span based on number of events: ' , args.span)

            success = 0
            cnt = 0
            success_list = []
            for snapshot in range(0, traj.frames-args.span, args.span//2):
                print('\n')
                cnt += 1
                if traj.save_frame[snapshot] and traj.save_frame[snapshot+args.span]:
                    if snapshot-args.span>=0 and not traj.save_frame[snapshot-args.span]:
                        print(colored('irregular data ignored, frame','red'), snapshot, ' -> ', snapshot+args.span)
                        continue
                      
                    print('save frame %d -> %d'%(snapshot, snapshot+args.span))
                    hg = traj.states[snapshot]
                    hg.span = args.span
                    event_list = set.union(*traj.edge_events[snapshot+1:snapshot+args.span+1])
                    elim_list = []
                    for checkpoint in range(snapshot+1, snapshot+args.span+1):
                        if len(traj.grain_events[checkpoint])>0:
                            for grain in traj.grain_events[checkpoint]:
                                elim_list.append([grain-1, args.span/(checkpoint-snapshot)])
                                
                    hg.form_gradient(prev = None if snapshot-args.span<0 else traj.states[snapshot-args.span], \
                                     nxt = traj.states[snapshot+args.span], event_list = event_list, elim_list = elim_list)
                    train_samples.append(hg)
                    
                    success_list.append(cnt)
                    success += 1
                    
                else:
                    print(colored('irregular data ignored, frame','red'), snapshot, ' -> ', snapshot+args.span)
                    
            print('sucess cases: ', success)        
        
        
            for idx, hg in enumerate(train_samples):

                frame = success_list[idx]                

                prev_list = []
                prev_idx_list = []
                
                for i in range(1, args.prev+1):
                    if frame - i in success_list:
                        prev_idx = success_list.index(frame-i)
                        prev_list.append(train_samples[prev_idx])
                        prev_idx_list.append(frame-i)
                    else:
                        prev_list.append(None)
                        prev_idx_list.append(-1)
                print('history list for %dth graph'%frame, prev_idx_list)
                hg.append_history(prev_list)
        
            with open(args.save_dir + 'seed' + str(seed) + '_G' + G + '_R' + R +\
                      '_edgeE' + edgeE + '_grainE' + grainE + '_span' + str(args.span) + '.pkl', 'wb') as outp:
                dill.dump(train_samples, outp)



    if args.mode == 'test':   
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir) 
        
    # creating testing dataset
        for seed in [args.seed]:
            
            test_samples = []
            
            traj = graph_trajectory(lxd=args.lxd, seed = seed, frames = args.frame, BC = args.boundary,
                                    adjust_grain_size = args.adjust_grain_size, 
                                    adjust_grain_orien = args.adjust_grain_orien)
            traj.match_graph = False
            traj.load_trajectory(rawdat_dir = args.rawdat_dir)
            hg0 = traj.states[0]

            with open('GR_train_grid.pkl', 'rb') as inp:  
                try:
                    GR_grid = dill.load(inp)
                except:
                    raise EOFError
            
            G_ = (traj.physical_params['G'] - GR_grid['G_min'])/(GR_grid['G_max'] - GR_grid['G_min'])
            R_ = (traj.physical_params['R'] - GR_grid['R_min'])/(GR_grid['R_max'] - GR_grid['R_min'])
            hg0.span = griddata(np.array([GR_grid['G'], GR_grid['R']]).T, np.array(GR_grid['span']), (G_, R_), method='nearest')
            
           # hg0.span = args.span
            hg0.form_gradient(prev = None, nxt = None, event_list = None, elim_list = None)
            hg0.append_history([])
            test_samples.append(hg0)
            
            G = str(round(traj.physical_params['G'],3))
            R = str(round(traj.physical_params['R'],3))


            with open(args.save_dir + 'seed' + str(seed) + '_G' + G + '_R' + R +\
                      '_span' + str(hg0.span) + '.pkl', 'wb') as outp:
                dill.dump(test_samples, outp)
                
            if args.save_traj:
                    with open(args.save_dir + 'traj' + str(seed) + '.pkl', 'wb') as outp:
                        dill.dump(traj, outp)   

    if args.mode == 'generate':   
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir) 
        
    # creating testing dataset
        for seed in [args.seed]:
            
            test_samples = []
            
            traj = graph_trajectory(lxd=args.lxd, seed = seed, frames = args.frame, physical_params = {'G':args.G, 'R':args.R},
                                    adjust_grain_size = args.adjust_grain_size,
                                    adjust_grain_orien = args.adjust_grain_orien)
            cur_grain, counts = np.unique(traj.alpha_field, return_counts=True)
            traj.area_counts = dict(zip(cur_grain, counts))
            traj.area_traj.append(traj.area_counts)            
            traj.form_states_tensor(0)

            hg0 = traj.states[0]

            with open('GR_train_grid.pkl', 'rb') as inp:  
                try:
                    GR_grid = dill.load(inp)
                except:
                    raise EOFError
            
            G_ = (traj.physical_params['G'] - GR_grid['G_min'])/(GR_grid['G_max'] - GR_grid['G_min'])
            R_ = (traj.physical_params['R'] - GR_grid['R_min'])/(GR_grid['R_max'] - GR_grid['R_min'])
            hg0.span = griddata(np.array([GR_grid['G'], GR_grid['R']]).T, np.array(GR_grid['span']), (G_, R_), method='nearest')
            
           # hg0.span = args.span
            hg0.form_gradient(prev = None, nxt = None, event_list = None, elim_list = None)
            hg0.append_history([])
            test_samples.append(hg0)
            
            G = str(round(traj.physical_params['G'],3))
            R = str(round(traj.physical_params['R'],3))


            with open(args.save_dir + 'seed' + str(seed) + '_G' + G + '_R' + R +\
                      '_span' + str(hg0.span) + '.pkl', 'wb') as outp:
                dill.dump(test_samples, outp)
                
            if args.save_traj:
                    with open(args.save_dir + 'traj' + str(seed) + '.pkl', 'wb') as outp:
                        dill.dump(traj, outp)                         
                        
    if args.mode == 'check':
        seed = 220
      #  g1 = graph(lxd = 20, seed=1) 
      #  g1.show_data_struct()
        traj = graph_trajectory(seed = seed, frames = 120)
        traj.load_trajectory(rawdat_dir = args.rawdat_dir)
    

  
# physical_params={'G':float(args.G), 'R':float(args.R)}
