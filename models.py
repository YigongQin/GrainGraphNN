#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 11:34:53 2021

@author: yigongqin
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from heteropgclstm import HeteroPGCLSTM, HeteroPGC
from heterogclstm import HeteroGCLSTM, HeteroGC
from graph_datastruct import periodic_move_p

class GC(nn.Module):


    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        num_layers: Number of LSTM layers stacked on each other
        bias: Bias or no bias in Convolution
        batch_first: Whether or not dimension 0 is the batch or not
        return_all_layers: Return the list of computations for all layers
    Input:
        A list of heterogeneous graph objects of length number of input sequence 
    Output:
        last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> nn = SeqGCLSTM(64, 16, 3,  True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, in_channels_dict, out_channels, num_layers, metadata, device,
                 bias=True, return_all_layers=True):
        super().__init__()

        #self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
     #   kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        out_channels = self._extend_for_multilayer(out_channels, num_layers)
        if not len(out_channels) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.in_channels_dict = in_channels_dict
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.metadata = metadata
        self.device = device
        self.bias = bias
        self.return_all_layers = return_all_layers
       # self.device = device

        cell_list = []
        for i in range(0, self.num_layers):
            
            if i == 0: 
                cur_in_channel = self.in_channels_dict  
                
                cell_list.append(HeteroPGC(in_channels_dict = cur_in_channel,
                                              out_channels = self.out_channels[i],
                                              metadata = self.metadata,
                                              bias = self.bias,
                                              device = self.device))                
            else:
                
                cur_in_channel = {node_type: self.out_channels[i - 1]
                                   for node_type in self.in_channels_dict}

                cell_list.append(HeteroGC(in_channels_dict = cur_in_channel,
                                              out_channels = self.out_channels[i],
                                              metadata = self.metadata,
                                              bias = self.bias,
                                              device = self.device))
                                     

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x_dict, edge_index_dict):

       
        """
        Arg types:
            * **x_dict** a list of x at different time, currently just one time
            * **edge_index_dict** a list of edge_index at different time
            * **hidden_state** a list of [h, c] pair of different layers,
                 each h, c is a dict of different node but with the same dimension
                 out_channels
        Return types:
            * **last_state_list** the same dimension of hidden_states
        
        """

        last_state_list = []

        cur_layer_x = x_dict
        for layer_idx in range(self.num_layers):
            
        


            h = self.cell_list[layer_idx](x_dict = cur_layer_x, \
                                             edge_index_dict = edge_index_dict)


            
            cur_layer_x = h
      
            last_state_list.append(h) # add [h, c] to current output layer 
            

        if not self.return_all_layers:
     
            last_state_list = last_state_list[-1:]

        return last_state_list # omit the hidden and cell state at intermediate time


    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param   
    

class SeqGCLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        num_layers: Number of LSTM layers stacked on each other
        bias: Bias or no bias in Convolution
        batch_first: Whether or not dimension 0 is the batch or not
        return_all_layers: Return the list of computations for all layers
    Input:
        A list of heterogeneous graph objects of length number of input sequence 
    Output:
        last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> nn = SeqGCLSTM(64, 16, 3,  True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, in_channels_dict, out_channels, num_layers, metadata, device,
                 bias=True, return_all_layers=True):
        super().__init__()

        #self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
     #   kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        out_channels = self._extend_for_multilayer(out_channels, num_layers)
        if not len(out_channels) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.in_channels_dict = in_channels_dict
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.metadata = metadata
        self.device = device
        self.bias = bias
        self.return_all_layers = return_all_layers
       # self.device = device

        cell_list = []
        for i in range(0, self.num_layers):
            
            if i == 0: 
                cur_in_channel = self.in_channels_dict  
                
                cell_list.append(HeteroPGCLSTM(in_channels_dict = cur_in_channel,
                                              out_channels = self.out_channels[i],
                                              metadata = self.metadata,
                                              bias = self.bias,
                                              device = self.device))                
            else:
                
                cur_in_channel = {node_type: self.out_channels[i - 1]
                                   for node_type in self.in_channels_dict}

                cell_list.append(HeteroGCLSTM(in_channels_dict = cur_in_channel,
                                              out_channels = self.out_channels[i],
                                              metadata = self.metadata,
                                              bias = self.bias,
                                              device = self.device))
                                     

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x_dict, edge_index_dict, hidden_state):

       
        """
        Arg types:
            * **x_dict** a list of x at different time, currently just one time
            * **edge_index_dict** a list of edge_index at different time
            * **hidden_state** a list of [h, c] pair of different layers,
                 each h, c is a dict of different node but with the same dimension
                 out_channels
        Return types:
            * **last_state_list** the same dimension of hidden_states
        
        """


        seq_len = 1  # change to seq2seq later

        if hidden_state is None:
            hidden_state = self._init_hidden(x_dict)


        last_state_list = []

         
        cur_layer_x = [x_dict]
        cur_layer_edge = [edge_index_dict]

        for layer_idx in range(self.num_layers):
            
            
            h, c = hidden_state[layer_idx] # initializa every layer to form H0, C0
          # output_inner = []
            layer_output = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](x_dict = cur_layer_x[t],
                                                 edge_index_dict = cur_layer_edge[t],
                                                 h_dict = h,
                                                 c_dict = c)
                ## output of each cell are dicts h and c 
    
              # output_inner.append(h)
                
         
                layer_output.append(h) # append each time output to a list 
                
                
 
           # layer_output = torch.stack(output_inner, dim=1) # stack in time dimension
            
            cur_layer_x = layer_output # the output of previous layer is the input
                                       # of next layer, connectivity remains
      
            last_state_list.append([h, c]) # add [h, c] to current output layer 
            

        if not self.return_all_layers:
     
            last_state_list = last_state_list[-1:]

        return last_state_list # omit the hidden and cell state at intermediate time

    def _init_hidden(self, x_dict):
        ## init the hidden states for every layer
        init_states = []
        for i in range(self.num_layers):
            h = self.cell_list[i]._set_hidden_state(x_dict, None)
            c = self.cell_list[i]._set_hidden_state(x_dict, None)
            init_states.append([h, c])
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    
    


    
    
class GrainNN_regressor(nn.Module):
    """ GrainNN in 3D
        Args:
            hyper: hyper-parameter class
    """
    
    def __init__(self, hyper):
        super().__init__()
  
        self.in_channels_dict = {node_type: len(features) 
                                 for node_type, features in hyper.features.items()}
        
        self.out_channels = hyper.layer_size 
        self.num_layer = hyper.layers
        self.metadata = hyper.metadata  # heterogeneous graph structure
        self.out_win = hyper.out_win
        
        self.device = hyper.device
      #  self.c = hyper.channel_table 

        ## networks

        self.gclstm_encoder = SeqGCLSTM(self.in_channels_dict, self.out_channels,\
                                        self.num_layer, self.metadata, self.device)
        self.gclstm_decoder = SeqGCLSTM(self.in_channels_dict, self.out_channels, \
                                        self.num_layer, self.metadata, self.device)
      #  self.gclstm_decoder2 = SeqGCLSTM(self.in_channels_dict, self.out_channels, \
      #                                  self.num_layer, self.metadata, self.device)
    
      #  self.gc_encoder = GC(self.in_channels_dict, self.out_channels,\
      #                                  self.num_layer, self.metadata, self.device)

        self.linear = nn.ModuleDict({node_type: nn.Linear(self.out_channels, len(targets))
                        for node_type, targets in hyper.targets.items()}) 

      #  self.GR_fit = hyper.GR_fit

    def forward(self, x_dict, edge_index_dict):
        
        """
        Making a forward pass.
        Arg types:
            * **x_dict** *(Dictionary where keys=Strings and values=PyTorch Float Tensors)* - Node features dicts. Can
                be obtained via PyG method :obj:`snapshot.x_dict` where snapshot is a single HeteroData object.
            * **edge_index_dict** *(Dictionary where keys=Tuples and values=PyTorch Long Tensors)* - Graph edge type
                and index dicts. Can be obtained via PyG method :obj:`snapshot.edge_index_dict`.

        Return types:
            * **y_dict** *(Dictionary where keys=Strings and values=PyTorch Float Tensor)* - Node type and
                output channels.
        """        

        hidden_state = self.gclstm_encoder(x_dict, edge_index_dict, None) # all layers of [h, c]
        
       # hidden_state = self.gc_encoder(x_dict, edge_index_dict) 
        
       # for i in range(self.out_win):
            
        hidden_state = self.gclstm_decoder(x_dict, edge_index_dict, hidden_state)
       # hidden_state = self.gclstm_decoder2(x_dict, edge_index_dict, hidden_state)
        
        h_dict, c_dict = hidden_state[-1]
       # h_dict = hidden_state[-1]    
        y_dict = {node_type: self.linear[node_type](h)
             for node_type, h in h_dict.items()} # apply the linear layer
        
        """
        
        GrainNN specific regressor output 
        
        """
     #   dx_scale = self.GR_fit[0] + self.GR_fit[1]*x_dict['joint'][:, 3] + self.GR_fit[2]*x_dict['joint'][:, 4]  
        
        #y_dict['joint'] = dx_scale.view(-1, 1)*torch.tanh(y_dict['joint']) # dx, dy are in the range [-1, 1]
        y_dict['joint'] = torch.tanh(y_dict['joint'])

        area = torch.tanh(y_dict['grain'][:, 0]) + x_dict['grain'][:, 3] # darea + area_old is positive  
        y_dict.update({'grain_event': torch.where(area<1e-6)[0] })
        
        area = F.relu(area)
       # area = F.normalize(area, p=1, dim=-1)
        y_dict['grain'][:, 0] = area - x_dict['grain'][:, 3]
 
        y_dict['grain'][:, 1] = F.relu(y_dict['grain'][:, 1]) # excess volume predict
        
                
        

            

        return y_dict            



    
    @staticmethod
    def update(x_dict, y_dict):
        
        # features
        x_dict['joint'][:, :2]  += y_dict['joint']
        x_dict['grain'][:, 3]   += y_dict['grain'][:, 0]
        x_dict['grain'][:, 4]   =  y_dict['grain'][:, 1]
        
        # gradients
        x_dict['joint'][:, -2:] =  y_dict['joint']
        x_dict['grain'][:, -1]  =  y_dict['grain'][:, 0]
        
        return x_dict


class GrainNN_classifier(torch.nn.Module):

    def __init__(self, hyper, regressor=None):
        super().__init__()
  
        self.in_channels_dict = {node_type: len(features) 
                                 for node_type, features in hyper.features.items()}
        
        self.out_channels = hyper.layer_size 
        self.num_layer = hyper.layers
        self.metadata = hyper.metadata  # heterogeneous graph structure
        self.out_win = hyper.out_win
        self.device = hyper.device

        ## networks
        if regressor:
            self.gclstm_encoder = regressor.gclstm_encoder
            self.gclstm_decoder = regressor.gclstm_decoder
            
        else:
            
            self.gclstm_encoder = SeqGCLSTM(self.in_channels_dict, self.out_channels,\
                                            self.num_layer, self.metadata, self.device)
            self.gclstm_decoder = SeqGCLSTM(self.in_channels_dict, self.out_channels, \
                                            self.num_layer, self.metadata, self.device)


        self.lin1 = nn.Linear(2*self.out_channels, 2) # predict dx, dy
        self.lin2 = nn.Linear(2*self.out_channels, 1) # predict probability
        self.threshold = 1
        
    def forward(self, x_dict, edge_index_dict):    
    

        hidden_state = self.gclstm_encoder(x_dict, edge_index_dict, None) # all layers of [h, c]
            
        hidden_state = self.gclstm_decoder(x_dict, edge_index_dict, hidden_state)
        
        h_dict, c_dict = hidden_state[-1]
        
        
        """
        
        GrainNN specific topological event
        
        """            
        

        joint_feature = h_dict['joint']
        joint_edge_index = edge_index_dict['joint', 'connect', 'joint']
        
        src, dst = joint_edge_index[0], joint_edge_index[1]

        # concatenate features [h_i, h_j], size (|Ejj|, 2*Dh)

        pair_feature = torch.cat([joint_feature[src], joint_feature[dst]], dim=-1)
  
           
        y_dict = {'edge_event': self.lin2(pair_feature).view(-1)} # p(i,j), size (Ejj,)

        y_dict['edge_rotation'] = torch.sigmoid(self.lin1(pair_feature)) - 0.5

        return y_dict


    def update(self, x_dict, edge_index_dict, y_dict):            
            

        E_pp = edge_index_dict['joint', 'connect', 'joint']
        E_qp = edge_index_dict['grain', 'push', 'joint']

        src, dst = E_pp[0], E_pp[1]
        
        prob = torch.sigmoid(y_dict['edge_event'])

        check_arg = [] #[45,67,134] #[45,67,78,112,125,134]
      #  print(joint_edge_index[:,check_arg])        
      #  z[check_arg] = 0.9
        ## predict eliminated edge
        L1 = ((prob>self.threshold)&(src<dst)).nonzero(as_tuple=True)
        

        """
        E2: Grain elimination
        """
        for grain in y_dict['grain_event']:
            Np = (E_qp[0]==grain).nonzero().view(-1)
            pairs = torch.combinations(Np, r=2)
            L2 = []
            for p1, p2 in pairs:
                if torch.tensor([p1, p2]) in E_pp and p1<p2:
                    E_index = ((E_pp[0]==p1)&(E_pp[1]==p2)).nonzero().view(-1)
                    L2.append(E_index)
                    if E_index in L1:
                        L1 = L1[L1!=E_index]
            L2 = torch.cat(L2)
            edge_index_dict = self.from_next_edge_index(edge_index_dict, x_dict, prob, L2, truncate=2)
            
        """
        E1: Neigbor switching
        """

        edge_index_dict = self.from_next_edge_index(edge_index_dict, x_dict, prob, L1)
                                
        return x_dict, edge_index_dict
    
    
    @staticmethod
    def from_next_edge_index(edge_index_dict, x_dict, prob, elimed_arg, truncate=0):

        print(elimed_arg)
        
        
      #  src, dst = edge_event_list
        E_pp = edge_index_dict['joint', 'connect', 'joint']
        E_pq = edge_index_dict['joint', 'pull', 'grain']
        E_qp = edge_index_dict['grain', 'push', 'joint']
        
        
      #  src_edge = src[elimed_arg]
      #  dst_edge = dst[elimed_arg]
        pairs = E_pp.T
        pairs = pairs[elimed_arg]
        ## sort
        sorted_prob, indices= torch.sort(prob[elimed_arg], dim=0, descending=True)
        #  print(p, elimed_prob)
        
      #  src_edge  = src_edge[indices]
      #  dst_edge  = dst_edge[indices]     
        pairs = pairs[indices][:-truncate] 
        
        
        
        for p1, p2 in pairs:
            
            # grain neighbors
            p1_qn_index = (E_pq[0]==p1).nonzero().view(-1)
            p1_qn = E_pq[1][ p1_qn_index ]
            p2_qn_index = (E_pq[0]==p2).nonzero().view(-1)
            p2_qn = E_pq[1][ p2_qn_index ]
            
            # joint neighbors
            p1_pn_index = ((E_pp[0]==p1)&(E_pp[1]!=p2)).nonzero().view(-1)
            p1_pn = E_pp[1][ p1_pn_index ]
            p2_pn_index = ((E_pp[0]==p2)&(E_pp[1]!=p1)).nonzero().view(-1)
            p2_pn = E_pp[1][ p2_pn_index ]            
            
            
            
            """
            p1_p2 = ((E_pp[0]==p1)&(E_pp[1]==p2)).nonzero().view(-1)
            p2_p1 = ((E_pp[0]==p2)&(E_pp[1]==p1)).nonzero().view(-1)
            
            x_dict['joint'][p1, :2]  = x_dict['joint'][p1, :2] - y_dict['joint'][p1] + y_dict['edge_rotation'][p1_p2]
            x_dict['joint'][p2, :2]  = x_dict['joint'][p2, :2] - y_dict['joint'][p2] + y_dict['edge_rotation'][p2_p1]
            
            x_dict['joint'][p1, -2:] =  y_dict['edge_rotation'][p1_p2]
            x_dict['joint'][p2, -2:] =  y_dict['edge_rotation'][p2_p1]
            
            """
            
            
            
            
            # find two expanding grains and two shrinking grains
            expand_q1 = p1_qn[(1-sum(p1_qn==i for i in p2_qn)).nonzero(as_tuple=True)] # new neighbor for p2
            expand_q2 = p2_qn[(1-sum(p2_qn==i for i in p1_qn)).nonzero(as_tuple=True)] # new neighbor for p1
        
            shrink_q1, shrink_q2 = p1_qn[(sum(p1_qn==i for i in p2_qn)).nonzero(as_tuple=True)]

            # swap the order
            p1_qn_index_sort = [p1_qn_index[i] for i in range(3) if p1_qn[i]==shrink_q1 ] + \
                               [p1_qn_index[i] for i in range(3) if p1_qn[i]==shrink_q2 ]
            p2_qn_index_sort = [p2_qn_index[i] for i in range(3) if p2_qn[i]==shrink_q1 ] + \
                               [p2_qn_index[i] for i in range(3) if p2_qn[i]==shrink_q2 ]
            
          #  print(p1_qn, p2_qn)
          #  print(expand_q1, expand_q2, shrink_q1, shrink_q2)
            # find two joints for one grain, they should be a pair for one joint

            
           # if torch.tensor([p1_pn[0], shrink_q1]) not in E_pq.T:
            if len(((E_pq[0]==p1_pn[0])&(E_pq[1]==shrink_q1)).nonzero().view(-1))>0:
                p1_pn = [p1_pn[0], p1_pn[1]]
                p1_pn_index = [p1_pn_index[0], p1_pn_index[1]]
            
            else:
                p1_pn = [p1_pn[1], p1_pn[0]]
                p1_pn_index = [p1_pn_index[1], p1_pn_index[0]]

            
           # if torch.tensor([p2_pn[0], shrink_q1]) not in E_pq.T:
            if len(((E_pq[0]==p2_pn[0])&(E_pq[1]==shrink_q1)).nonzero().view(-1))>0:
                p2_pn = [p2_pn[0], p2_pn[1]]
                p2_pn_index = [p2_pn_index[0], p2_pn_index[1]]
                
            else:
                p2_pn = [p2_pn[1], p2_pn[0]]
                p2_pn_index = [p2_pn_index[1], p2_pn_index[0]]   

            sq1_p1, sq2_p1 = p1_pn
            sq1_p2, sq2_p2 = p2_pn
            
            
          #  print('\n',p1_pn, p2_pn)
          #  print(p1_pn_index, p2_pn_index)


            if point_in_triangle(x_dict['joint'][p2,:2], x_dict['joint'][p1,:2], \
                                 x_dict['joint'][sq1_p1,:2], x_dict['joint'][sq1_p2,:2]):  
               # print('swap')
                # swap
                p1_qn_index_sort.reverse()
                p2_qn_index_sort.reverse()
                p1_pn_index.reverse()
                p2_pn_index.reverse()
                
                sq1_p1, sq2_p1 = sq2_p1, sq1_p1
                sq1_p2, sq2_p2 = sq2_p2, sq1_p2

                
            # replace joint-grain edges
          #  print(p1_qn_index_sort, p2_qn_index_sort)
          #  print(E_pq[1, p1_qn_index_sort[1]], expand_q2)
          #  print(E_pq[1, p2_qn_index_sort[0]], expand_q1)
            E_pq[1, p1_qn_index_sort[1]] = expand_q2 # reject sq2
            E_pq[1, p2_qn_index_sort[0]] = expand_q1
            
         #   print(E_pp[1, p1_pn_index[1]], sq1_p2)
         #   print(E_pp[1, p2_pn_index[0]], sq2_p1)            
            # replace joint-joint edges
            E_pp[1, p1_pn_index[1]] = sq1_p2 # reject sq2_p1
            E_pp[1, p2_pn_index[0]] = sq2_p1
            
           # print((sq1_p2, p2),'->', (sq1_p2, p1))
           # print((sq2_p1, p1),'->', (sq2_p1, p2))              
            E_pp[1][ ((E_pp[0]==sq1_p2)&(E_pp[1]==p2)).nonzero().view(-1) ] = p1
            E_pp[1][ ((E_pp[0]==sq2_p1)&(E_pp[1]==p1)).nonzero().view(-1) ] = p2
      
            
            
        E_qp[0], E_qp[1] = E_pq[1], E_pq[0]       
        
        return edge_index_dict


def point_in_triangle(t, v1, v2, v3):
    sign = lambda a, b, c: (a[0] - c[0])*(b[1] - c[1]) - \
                           (b[0] - c[0])*(a[1] - c[1])
   # print(v1, t)
    periodic_move_p(v1, t)
    periodic_move_p(v2, t)
    periodic_move_p(v3, t)
  #  print(t, v1, v2, v3)
    d1 = sign(t, v1, v2)
    d2 = sign(t, v2, v3)
    d3 = sign(t, v3, v1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg & has_pos)


     

"""

  self.features = {'grain':['x', 'y', 'z', 'area', 'extraV', 'cosx', 'sinx', 'cosz', 'sinz'],
                   'joint':['x', 'y', 'z', 'G', 'R']}
  
  self.features_grad = {'grain':['darea'], 'joint':['dx', 'dy']}
  
  self.targets = {'grain':['darea', 'extraV'], 'joint':['dx', 'dy']}    

  self.edge_type = [('grain', 'push', 'joint'), \
                      ('joint', 'pull', 'grain'), \
                      ('joint', 'connect', 'joint')]
            
"""

"""
def rot90(data, symmetry, rot):
    
    if rot == 0: return
    
    signx, signy = -1, -1
 #   signx = 
    

    if symmetry == 2:
        data.x_dict['grain'][:,:2]  = 1 - data.x_dict['grain'][:,:2]
        data.x_dict['grain'][:,5:6] = -data.x_dict['grain'][:,5:6]
        data.x_dict['joint'][:,:2]  = 1 - data.x_dict['joint'][:,:2]
        data.x_dict['joint'][:,-2:] = -data.x_dict['joint'][:,-2:]
        data.y_dict['joint'][:,:2]  = -data.y_dict['joint'][:,:2]
 
    if symmetry == 4:
        
 
    
        data.x_dict['grain'][:,0], data.x_dict['grain'][:,1]   = 1 - data.x_dict['grain'][:,1], data.x_dict['grain'][:,0]
        data.x_dict['grain'][:,5], data.x_dict['grain'][:,6]   = data.x_dict['grain'][:,6], -data.x_dict['grain'][:,5]    
        
        data.x_dict['joint'][:,0], data.x_dict['joint'][:,1]   = 1 - data.x_dict['joint'][:,1], data.x_dict['joint'][:,0]    
        data.x_dict['joint'][:,-2], data.x_dict['joint'][:,-1] = -data.x_dict['joint'][:,-1], data.x_dict['joint'][:,-2] 
        
        data.y_dict['joint'][:,0], data.y_dict['joint'][:,1]   = -data.y_dict['joint'][:,1], data.y_dict['joint'][:,0]  
        
"""
