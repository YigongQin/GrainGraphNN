#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 14:55:03 2022

@author: yigongqin
"""

import numpy as np
import h5py
import glob, re, os, argparse, dill
from collections import defaultdict
from termcolor import colored
import matplotlib.pyplot as plt
import itertools
from graph_datastruct import graph, GrainHeterograph, periodic_move
from math import pi


def relative_angle(p1, p2):
    
    p1 = periodic_move(p1, p2)
    
    return np.arctan2(p2[1]-p1[1], p2[0]-p1[0])


class QoI_trajectory:
    def __init__(self):
        
        self.num_grains
        
    def plot(self):
        
        return



class graph_trajectory(graph):
    def __init__(self, lxd: float = 20, seed: int = 1, noise = 0.01, frames: int = 1, physical_params = {}):   
        super().__init__(lxd = lxd, seed = seed, noise = noise)
        
        self.joint2vertex = dict((tuple(sorted(v)), k) for k, v in self.vertex2joint.items())
        self.frames = frames # note that frames include the initial condition
        self.joint_traj = []
        self.edge_events = []
        self.grain_events = []
        
        self.show = False
        self.states = []
        self.physical_params = physical_params
        self.save_frame = [True]*self.frames

    def load_trajectory(self, rawdat_dir: str = './'):
       
        
        self.data_file = (glob.glob(rawdat_dir + '/*seed'+str(self.seed)+'_*'))[0]
        f = h5py.File(self.data_file, 'r')
        self.x = np.asarray(f['x_coordinates'])
        self.y = np.asarray(f['y_coordinates'])
        self.z = np.asarray(f['z_coordinates']) 

        assert int(self.lxd) == int(self.x[-2])
        
        self.x /= self.lxd; self.y /= self.lxd; self.z /= self.lxd
        fnx, fny = len(self.x), len(self.y)

        assert len(self.x) -2 == self.imagesize[0]
        assert len(self.y) -2 == self.imagesize[1]  
        
        number_list=re.findall(r"[-+]?\d*\.\d+|\d+", self.data_file)
        data_frames = int(number_list[2])+1
        
        self.physical_params = {'G':float(number_list[3]), 'R':float(number_list[4])}
        self.alpha_pde_frames = np.asarray(f['cross_sec'])
        self.alpha_pde_frames = self.alpha_pde_frames.reshape((fnx, fny, data_frames),order='F')[1:-1,1:-1,:]        
        
        self.extraV_frames = np.asarray(f['extra_area'])
        self.extraV_frames = self.extraV_frames.reshape((self.num_regions, data_frames), order='F')        
     
        self.num_vertex_features = 8  ## first 2 are x,y coordinates, next 5 are possible phase
        self.active_args = np.asarray(f['node_region'])
        self.active_args = self.active_args.\
            reshape((self.num_vertex_features, 5*len(self.vertices), data_frames ), order='F')
        self.active_coors = self.active_args[:2,:,:]
        self.active_max = self.active_args[2,:,:]
        self.active_args = self.active_args[3:,:,:]
        

        prev_joint = {k:[0,0,100] for k, v in self.joint2vertex.items()}
        all_grain = set(np.arange(self.num_regions)+1)
        
        for frame in range(self.frames):
           
            
            print('load frame %d'%frame)
            
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

            
            """
            deal with quadruples 
            
            """

            
            ## delete undetermined junctions from quadruples, add right ones later
            del_joints = []
            for q, coors in quadraples.items():
                q_list = list(q)
              #  print(q_list)
                for comb in [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]:
                    arg = tuple([q_list[i] for i in comb])

                    if arg not in prev_joint and arg in cur_joint:

                        del_joints.append([arg, cur_joint[arg]])
                        del cur_joint[arg]
                        
            print('quadruples', len(quadraples))
            
            def check_connectivity(cur_joint):
               # jj_link = 0
               
                missing = set()
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

                        missing.update(set(k1))
                        miss_case.update({k1:3-num_link})
                        total_missing += abs(3-num_link)
                     #   print('find missing junction link', k1, 3-num_link)
                return total_missing, missing, miss_case   

            def miss_quadruple(quadraples):

                total_missing, missing, miss_case  = check_connectivity(cur_joint)
                print('total missing edges, ', total_missing)
                for q, coor in quadraples.items():
                    if set(q).issubset(missing):
                        
                        print('using quadraples',q,' to find missing link')
                        
                        possible = list(itertools.combinations(list(q), 3))
                        for c in miss_case.keys():
                            if c in possible:
                                possible.remove(c)
    
                        miss_case_sum = 0

                        for i, j in miss_case.items():
                            if len(set(i).intersection(set(q)))>=2:
                                miss_case_sum += j
                             #   max_case = max(max_case, j)                 
                            
                        print('np. missing links', miss_case_sum)        
                        max_case = 1 if miss_case_sum<4 else 2
                        for ans in list(itertools.combinations(possible, max_case)):
                            print('try ans', ans)
                            for a in ans:
                                cur_joint[a] = coor
                            cur, _, _ = check_connectivity(cur_joint)
                            if cur == total_missing -miss_case_sum:
                                print('fixed!')
                                total_missing = cur
                                break
                            else:
                                for a in ans:
                                        del cur_joint[a]
                                    
            miss_quadruple(quadraples)

            self.joint_traj.append(cur_joint)
            prev_joint = cur_joint
            


            # check loaded information
            
            self.alpha_pde = self.alpha_pde_frames[:,:,frame].T
            cur_grain, counts = np.unique(self.alpha_pde, return_counts=True)
            self.area_counts = dict(zip(cur_grain, counts))
            self.area_counts = {i:self.area_counts[i] if i in self.area_counts else 0 for i in range(self.num_regions)}
            cur_grain = set(cur_grain)
            
            
            grain_set = set()
            for k in cur_joint.keys():
                grain_set.update(set(k))



            eliminated_grains = all_grain - cur_grain
            
            


            if len(cur_joint)<2*len(cur_grain):
                for arg, coor in del_joints:
                    cur_joint[arg] = coor
                miss_quadruple(quadraples)
                
            
            if len(cur_joint)!=2*len(cur_grain):
                print(colored('junction find failed', 'red'))
                print(len(cur_joint), len(cur_grain))
                self.grain_events.append(set())
                self.edge_events.append(set())  
                self.form_states_tensor(frame)
                
               # exit()
                continue
            
            print('number of grains in pixels %d'%len(cur_grain))
        #    print('number of grains junction %d'%len(grain_set))
            print('number of junctions %d'%len(cur_joint))
            
            all_grain = cur_grain
            
          #  print('estimated number of junction-junction links %d'%jj_link) 
            # when it approaches the end, 3*junction is not accurate
            self.edge_labels = {(src, dst):0 for src, dst in self.edges}
            for grain in eliminated_grains:
                for pair in self.region_edge[grain]:
                    self.edge_labels[(pair[0], pair[1])] = -100
                    self.edge_labels[(pair[1], pair[0])] = -100                                 
            
            self.vertex_matching(frame, cur_joint, eliminated_grains)
        
            self.update()
            self.form_states_tensor(frame)
          #  if self.error_layer>0.08:
          #      self.save_frame[frame] = False
          #  if len(self.edges)!=6*len(cur_grain):
          #      self.save_frame[frame] = False
                
            if self.show == True:
                self.show_data_struct()   
                
            print('====================================')  
            print('\n') 
              
                
    def vertex_matching(self, frame, cur_joint, eliminated_grains):
      
      
        print('\n')
        print('summary of event from frame %d to frame %d'%(frame-1, frame))
      #  cur_joint = self.joint_traj[frame]


        def add_edge_event(old_junction_i, old_junction_j):
            vert_old_i = self.joint2vertex[old_junction_i]
            vert_old_j = self.joint2vertex[old_junction_j]                
            switching_edges.add((vert_old_i, vert_old_j))
            switching_edges.add((vert_old_j, vert_old_i))
            self.edge_labels[(vert_old_i, vert_old_j)] = 1
            self.edge_labels[(vert_old_j, vert_old_i)] = 1  


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
        
        
        def ns_last_vert(j1, i1, i2, q):

            j2 = set(q) - set(j1)
            j2.add(list(set(i1)-set(i2))[0])
            j2.add(list(set(i2)-set(i1))[0])

            return tuple(sorted(list(j2)))

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
            
            """
            l = list(self.edges.values())
            e_key = list(self.edges.keys())
            idx = l.index([vert_old_i, N_i[1]])
            self.edges[e_key[idx]] = [vert_old_i, N_j[1]]
            idx = l.index([vert_old_j, N_j[1]])
            self.edges[e_key[idx]] = [vert_old_j, N_i[1]]  
            
            idx = l.index([N_i[1], vert_old_i])
            self.edges[e_key[idx]] = [N_j[1], vert_old_i]
            idx = l.index([N_j[1], vert_old_j])
            self.edges[e_key[idx]] = [N_i[1], vert_old_j]  
            """
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
                    for neigh in self.vertex_neighbor[vert]:
                        if neigh not in old_vert:
                            for joint in toadd:
                                if len(set(joint).intersection(set(self.vertex2joint[neigh])))==2:
                                    # find new link
                                    if joint in visited_joint:
                                        remove_vert.append([vert, visited_joint[joint]])
                                    else:
                                        visited_joint.update({joint:vert})
                print(remove_vert)                        
                o1, o2 = remove_vert[0][0], remove_vert[1][0]
                r1, r2 = remove_vert[0][1], remove_vert[1][1]
              #  print(o1, o2, old_vert)  
                old_vert.remove(o1)
                old_vert.remove(o2)                        
                
                connect = len(set(self.vertex2joint[o1]).intersection(set(self.vertex2joint[o2])))==2
            
                for v1 in old_vert:
                    for v2 in old_vert:
                        if [v1, v2] in self.edges:
                            self.edges[self.edges.index([v1, v2])] = [-1, -1]
                            self.edges[self.edges.index([v2, v1])] = [-1, -1]
                            
                for k1 in toadd:
                    for k2 in toadd: 
                        if k1 != k2 and len( set(k1).intersection(set(k2)) ) == 2:
                            v1 = visited_joint[k1]
                            v2 = visited_joint[k2]
                            if [v1, v2] not in self.edges:
                                self.edges.append([v1, v2])
                                self.edges.append([v2, v1])     



                ''' remove vertices connect to elim grains '''     
                print(elm_grain,'th grain eliminated with no. of sides %d'%len(todelete), junction)
                for k in todelete:
                    del self.joint2vertex[k]       
                    

                ''' add vertices '''     
                for joint, vert in visited_joint.items():
                    self.joint2vertex[joint] = vert
                    print('the new joint', joint, 'inherit the vert', vert)

                '''  '''    

                
                       
                            
                
                def elim_edge(o1, o2, r1, r2):
                    N1 = [i for i, x in enumerate(self.edges) if x[1] == o1 ] 
                    
                    case = 1
                    for i in N1:
                        src = self.edges[i][0]

                        if src == o2:
                            self.edges[i] = [-1, -1]
                            
                        elif src in old_vert:
                            idx = self.edges.index([o1, src])    
                            self.edges[i] = [-1, -1]
                            
                          #  if not connect and case:
                          #      self.edges[idx] = [r1, r2]
                          #      case -= 1
                           # else: 
                            self.edges[idx] = [-1, -1]
                            
                        else:
                            idx = self.edges.index([o1, src]) 
                            
                            self.edges[i] = [src, r1]
                            self.edges[idx] = [r1, src]
                            
                            print(o1, src, 'replace by', r1, src)

                elim_edge(o1, o2, r1, r2)
                elim_edge(o2, o1, r2, r1)
         #   assert len(old_vert) == 2

            
        self.grain_events.append(eliminated_grains)
        self.edge_events.append(switching_edges)    
        

        for joint in self.joint2vertex.keys():
            if joint in cur_joint:
                vert = self.joint2vertex[joint]
                coors = cur_joint[joint]
                self.vertices[vert] = periodic_move(coors, old_vertices[vert])
      
            else:
                vert = self.joint2vertex[joint]
                self.vertices[vert] = old_vertices[vert]
                print(colored('unmatched joint detected: ', 'red'), joint, self.joint2vertex[joint])
        for joint in cur_joint.keys():
            if joint not in self.joint2vertex:
                print(colored('unused joint detected: ', 'green'), joint)
      
        
       
        print('number of E2 %d, number of E1 %d'%(len(eliminated_grains), len(switching_edges)//2))
        
      

            

            

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
        
        s = self.imagesize[0]
        

        
        for grain, coor in self.region_center.items():
            grain_state[grain-1, 0] = coor[0]
            grain_state[grain-1, 1] = coor[1]
            grain_mask[grain-1, 0] = 1

        grain_state[:, 2] = frame/self.frames

        grain_state[:, 3] = np.array(list(self.area_counts.values())) /s**2
        if frame>0:
            grain_state[:, 4] = self.extraV_frames[:, frame]/s**3
        
        
        grain_state[:, 5] = np.cos(self.theta_x[1:])
        grain_state[:, 6] = np.sin(self.theta_x[1:])
        grain_state[:, 7] = np.cos(self.theta_z[1:])
        grain_state[:, 8] = np.sin(self.theta_z[1:])
        
        
        for joint, coor in self.vertices.items():
            joint_state[joint, 0] = coor[0]
            joint_state[joint, 1] = coor[1]
            joint_mask[joint, 0] = 1
        
        joint_state[:, 2] = frame/self.frames
        joint_state[:, 3] = 1 - np.log10(self.physical_params['G'])/2
        joint_state[:, 4] = self.physical_params['R']/2
        
        
        gj_edge = []
        for grains, joint in self.joint2vertex.items():
            for grain in grains:
                gj_edge.append([grain-1, joint])
        
        jg_edge = [[joint, grain] for grain, joint in gj_edge]
        jj_edge = [[src, dst] for src, dst in self.edges]
        
        
        hg.feature_dicts.update({'grain':grain_state})
        hg.feature_dicts.update({'joint':joint_state})
        hg.edge_index_dicts.update({hg.edge_type[0]:np.array(gj_edge).T})
        hg.edge_index_dicts.update({hg.edge_type[1]:np.array(jg_edge).T})
        hg.edge_index_dicts.update({hg.edge_type[2]:np.array(jj_edge).T})
        
        
        hg.mask = {'grain':grain_mask, 'joint':joint_mask}
        
     #   joint_grain_neighbor = -np.ones((self.num_vertices,3), dtype=int)
      #  joint_joint_neighbor = -np.ones((self.num_vertices,3), dtype=int)
        for k, v in self.vertex_neighbor.items():
            if len(v)<3: print(colored('junction with less than three junction neighbor', 'red'), k)
            if len(v)>3: print(colored('junction with more than three junction neighbor', 'red'), k)
            assert len(v)==3
      #      joint_joint_neighbor[k][:len(v)] = np.array(list(v))
        
        hg.vertex2joint = self.vertex2joint
      #  for k, v in self.joint2vertex.items():
      #      joint_grain_neighbor[v] = np.array(list(k))
        
      #  hg.neighbor_dicts.update({('joint','joint'):joint_joint_neighbor})
      #  hg.neighbor_dicts.update({('joint','grain'):joint_grain_neighbor})
        
        hg.physical_params = self.physical_params
        hg.physical_params.update({'seed':self.seed})

        if frame>0:
            hg.edge_rotation = np.array(list(self.edge_labels.values()))

        
        self.states.append(hg) # states at current time

        # reset the edge_labels, check event at next snapshot
        # self.edge_labels = {(src, dst):0 for src, dst in jj_edge} 

if __name__ == '__main__':


    parser = argparse.ArgumentParser("Generate heterograph trajectory")
    parser.add_argument("--mode", type=str, default = 'check')
    parser.add_argument("--rawdat_dir", type=str, default = './')
    parser.add_argument("--train_dir", type=str, default = './sameGR/')
    parser.add_argument("--test_dir", type=str, default = './test/')
    parser.add_argument("--seed", type=int, default = 1)
    parser.add_argument("--level", type=int, default = 0)
    parser.add_argument("--frame", type=int, default = 13)
    args = parser.parse_args()
    args.train_dir = args.train_dir + 'level' + str(args.level) +'/'
    args.test_dir = args.test_dir + 'level' + str(args.level) +'/'
    """
    this script generates graph trajectory objects and training/testing data 
    the pde simulaion data is in rawdat_dir
    train_dir: processed data for training
    test_dir: processed data for testing
    seed: realization seed, each graph trajectory relates to one pde simulation
    level: 0: regression 
           1: regression + classification 
           2: regression + classification + mask 
    """
    
    
    
    if args.mode == 'train':
    
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
  
        for seed in [args.seed]:
            
            train_samples = []
            
            traj = graph_trajectory(seed = seed, frames = args.frame)
          #  traj.update()
          #  traj.show_data_struct()
      
            traj.load_trajectory(rawdat_dir = args.rawdat_dir)
            #traj.show_data_struct()
            
            with open(args.train_dir + 'traj' + str(seed) + '.pkl', 'wb') as outp:
                dill.dump(traj, outp)
                
            for snapshot in range(traj.frames-1):
                """
                training data: snapshot -> snapshot + 1
                whether data is useful depends on both 
                <1> regression part:
                    grain exists at snapshot
                    triple junction exists at both (shift only)
                    
                <2> classification part:
                    edge exists at both (label 0)
                    edge switching (label 1)
                    unknown (mask out)
                    
                """
                if traj.save_frame[snapshot+1] == True:
                    
                    if ( args.level == 2 ) \
                    or ( args.level == 1 and len(traj.grain_events[snapshot+1])==0 ) \
                    or ( args.level == 0 and len(traj.grain_events[snapshot+1])==0 and \
                                             len(traj.edge_events[snapshot+1])==0 ):    
                        
                        hg = traj.states[snapshot]
                        hg.form_gradient(prev = None if snapshot ==0 else traj.states[snapshot-1], \
                                         nxt = traj.states[snapshot+1])
                        print('save frame %d -> %d, event level %d'%(snapshot, snapshot+1, args.level))
                        train_samples.append(hg)
                else:
                    print(colored('irregular data ignored, frame','red'), snapshot+1)
       
            with open(args.train_dir + 'case' + str(seed) + '.pkl', 'wb') as outp:
                dill.dump(train_samples, outp)



    if args.mode == 'test':   
        if not os.path.exists(args.test_dir):
            os.makedirs(args.test_dir) 
        
    # creating testing dataset
        for seed in [1]:
            
            test_samples = []
            
            traj = graph_trajectory(seed = seed, frames = 1, physical_params={'G':5, 'R':1})
            traj.load_trajectory(rawdat_dir = args.rawdat_dir)
            hg0 = traj.states[0]
            hg0.form_gradient(prev = None, nxt = None)
            test_samples.append(hg0)
          #  hg0.graph = graph(seed = seed)
            with open(args.test_dir + 'case' + str(seed) + '.pkl', 'wb') as outp:
                dill.dump(test_samples, outp)
     
        
    if args.mode == 'check':
        seed = 0
      #  g1 = graph(lxd = 20, seed=1) 
      #  g1.show_data_struct()
        traj = graph_trajectory(seed = seed, frames = 40, noise=0.01)
        traj.load_trajectory(rawdat_dir = args.rawdat_dir)
    
    if args.mode == 'instance':
        
        for seed in range(20):
            print('\n')
            print('test seed', seed)
            try:
                g1 = graph(lxd = 20, seed=seed) 
            except:    
                print('seed %d failed with noise 0.01, try 0'%seed)
                g1 = graph(lxd = 20, seed=seed, noise = 0.0)

            g1.show_data_struct() 
            


            
            
"""
if left_over!= -1:
    for q, joints in quadraples_new.items():
        for i in old_joint:
            if set(i).issubset(set(q)):            
                add_vert = ns_last_vert(i, joints[0], joints[1], q)
                
                self.joint2vertex[add_vert] = left_over
                if i in old_joint: old_joint.remove(i)
                if joints[0] in new_joint: new_joint.remove(joints[0])
                if joints[1] in new_joint: new_joint.remove(joints[1])                           
                perform_switching(add_vert, i, joints[0], joints[1])



case = 0 
for q, joints in quadraples.items():
    for j in new_joint:
        if set(j).issubset(set(q)) and not case:
            add_vert = ns_last_vert(j, joints[0], joints[1], q)
           # print(old_joint, joints[0], joints[1])
            add_edge_event(joints[0], joints[1])
            perform_switching(joints[0], joints[1], j, add_vert)
            old_joint.append(add_vert)
            case = 1

quadraples = quadruple_(old_joint)
quadraples_new = quadruple_(new_joint)

switching_event = set(quadraples.keys()).intersection(set(quadraples_new.keys()))

for e2 in switching_event:
    
    old_junction_i, old_junction_j = quadraples[e2]
    new_junction_i, new_junction_j = quadraples_new[e2]

    perform_switching(old_junction_i, old_junction_j, new_junction_i, new_junction_j)    

            

            
            
if len(old_joint)>0:
    print(colored('match not finisehd','red'))

"""
            
            
        
"""
old_map, new_map = match()            
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
  
    '''remove vertices connect to elim grains'''  
    print(elm_grain,'th grain eliminated with no. of sides %d'%len(todelete), junction)
    for k in todelete:
        del self.joint2vertex[k]   
        
    
    for k, v in new_map.items():
        if set(k).issubset(junction):# and k not in self.joint2vertex:      
            toadd.append(k)                    
  
    
      #  left_over = old_vert[-1]
    diff = len(old_vert) - len(toadd) - 2
    neigh = []
    '''find missing vertices'''
    for k, v in new_map.items():
        if len( set(k).intersection(junction) ) == 2:
            neigh.append([periodic_dist_(self.region_center[elm_grain[0]], v), k])
            
    neigh = sorted(neigh)
    for i in range(diff):
        toadd.append(neigh[-i-1][1])
    
  #  print(toadd)
    ''' add vertices '''       
    for i in range(len(toadd)):
        self.joint2vertex[toadd[i]] = old_vert[i]
        print('the new joint', toadd[i], 'inherit the vert', old_vert[i])
        del new_map[toadd[i]]
    
"""            
