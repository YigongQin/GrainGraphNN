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
import copy

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

    def forward(self, x_dict, edge_index_dict, edge_attr, hidden_state):

       
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
        
        if hidden_state is None:
            hidden_state = self._init_hidden(x_dict)

        last_state_list = []

        cur_layer_x = x_dict
        for layer_idx in range(self.num_layers):
            
        
            h, c = hidden_state[layer_idx]

            h, c = self.cell_list[layer_idx](x_dict = cur_layer_x, \
                                             edge_index_dict = edge_index_dict,
                                             edge_attr = edge_attr,
                                             h_dict = h,
                                             c_dict = c)

            
            cur_layer_x = h
      
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

    def forward(self, x_dict, edge_index_dict, edge_attr, hidden_state):

       
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
                                                 edge_attr = edge_attr,
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
    
    
class LSTM(nn.Module):
    
    def __init__(self, in_channels_dict, out_channels, num_layers, device,
                 dim, seq_len = 1, bias = True, return_all_layers = False):
        super().__init__()
        
        self.in_channels_dict = in_channels_dict
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.device = device
        self.dim = dim
        self.seq_len = seq_len
        self.bias = bias
        self.return_all_layers = return_all_layers        
        
        
        self.nn = nn.ModuleDict({node_type: nn.LSTM(input_size=dim, hidden_size=self.out_channels, num_layers=2, batch_first=True) 
                        for node_type, dim in self.dim.items()}) 
       
        
    def forward(self, x_dict, hidden_state=None):
        


        dx_dict = {}
        for node_type, dim in self.dim.items():
            
            feature_list = []
            
            for i in range(dim):
                dx_i = x_dict[node_type][:,-self.seq_len*dim+i::dim]
                feature_list.append(torch.flip(dx_i, [1]))
                
            dx = torch.stack(feature_list, dim=2)
            dx_dict.update({node_type:dx})

       # dx_dict = {node_type: x_dict[node_type][:,-self.seq_len*dim:].view(-1, self.seq_len, dim)
       #      for node_type, dim in self.dim.items()}       
            
        y_dict = {node_type: self.nn[node_type](dx_dict[node_type])[0][:,-1,:]
             for node_type, dim in self.dim.items()}     
        
        
        return y_dict
        
        
    
class GrainNN_regressor(nn.Module):
    """ GrainNN in 3D
        Args:
            hyper: hyper-parameter class
    """
    
    def __init__(self, hyper, history=False, edge_len=False):
        super().__init__()
  
        self.in_channels_dict = {node_type: len(features) 
                                 for node_type, features in hyper.features.items()}
        
        self.out_channels = hyper.layer_size 
        self.num_layer = hyper.layers
        self.metadata = hyper.metadata  # heterogeneous graph structure
        self.out_win = hyper.out_win
        
        self.device = hyper.device
        self.seq_len = hyper.window
        self.history = history
        self.edge_len = edge_len
        ## networks

        self.gclstm_encoder = SeqGCLSTM(self.in_channels_dict, self.out_channels,\
                                        self.num_layer, self.metadata, self.device)
        self.gclstm_decoder = SeqGCLSTM(self.in_channels_dict, self.out_channels, \
                                        self.num_layer, self.metadata, self.device)
      #  self.gclstm_decoder2 = SeqGCLSTM(self.in_channels_dict, self.out_channels, \
      #                                  self.num_layer, self.metadata, self.device)
    
      #  self.gc_encoder = GC(self.in_channels_dict, self.out_channels,\
      #                                  self.num_layer, self.metadata, self.device)
        self.dim = {'joint':2, 'grain':1}
        
        if self.history:
            self.LSTM = LSTM(self.in_channels_dict, self.out_channels, self.num_layer, self.device,
                             self.dim, seq_len = self.seq_len)    
            
            
        linear_outchannels = 2*self.out_channels if self.history else self.out_channels
        
        self.linear = nn.ModuleDict({node_type: nn.Linear(linear_outchannels, len(targets))
                        for node_type, targets in hyper.targets.items()}) 
        
        if self.edge_len:
            self.lin1 = nn.Linear(2*linear_outchannels+2, 1)
        
        self.scaling = {'grain':20, 'joint':5}


    def forward(self, x_dict, edge_index_dict, edge_attr):
        
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

        if self.history:
            history_encoded = self.LSTM(x_dict)

      #  x_dict = {node_type: x_dict[node_type][:, :-(self.seq_len-1)*dim]
      #       for node_type, dim in self.dim.items()}  

        hidden_state = self.gclstm_encoder(x_dict, edge_index_dict, edge_attr, None) # all layers of [h, c]
            
        hidden_state = self.gclstm_decoder(x_dict, edge_index_dict, edge_attr, hidden_state)

        h_dict, c_dict = hidden_state[-1]


        if self.history:
            h_dict = {node_type: torch.cat([h, history_encoded[node_type]], dim=1)
                      for node_type, h in h_dict.items()}
        
        y_dict = {node_type: self.linear[node_type](h)
             for node_type, h in h_dict.items()} # apply the linear layer
        
        """
        
        GrainNN specific regressor output 
        
        """


        y_dict['joint'] = torch.tanh(y_dict['joint'])    # dx, dy are in the range [-1, 1]

        area = torch.tanh(y_dict['grain'][:, 0])/self.scaling['grain'] + x_dict['grain'][:, 3] # darea + area_old is positive  
        y_dict.update({'grain_area': area })
        
       # area = F.relu(area)
       # area = F.normalize(area, p=1, dim=-1)
        y_dict['grain'][:, 0] = torch.tanh(y_dict['grain'][:, 0]) #area - x_dict['grain'][:, 3]
 
        y_dict['grain'][:, 1] = F.relu(y_dict['grain'][:, 1]) # excess volume predict
        
                
        if self.edge_len: 
            joint_feature = h_dict['joint']
            joint_edge_index = edge_index_dict['joint', 'connect', 'joint']
            
            src, dst = joint_edge_index[0], joint_edge_index[1]
    
            pair_feature = torch.cat([joint_feature[src], joint_feature[dst], edge_attr['joint', 'connect', 'joint']], dim=-1)
    
    
            y_dict['edge'] = torch.tanh(self.lin1(pair_feature)) 
            

        return y_dict            



    

    def update(self, x_dict, y_dict):
        
        # features
        x_dict['joint'][:, :2]  += y_dict['joint']/self.scaling['joint']  # junction movement -> (x,y)
        x_dict['grain'][:, 3]   += y_dict['grain'][:, 0]/self.scaling['grain']  # grain area change -> s
        x_dict['grain'][:, 4]   =  y_dict['grain'][:, 1]  # extra volume v
        
        # gradients
        x_dict['joint'][:, 6:8] =  y_dict['joint']  # dx, dy
        x_dict['grain'][:, -1]  =  y_dict['grain'][:, 0]  # ds
        
       # return x_dict


class GrainNN_classifier(torch.nn.Module):

    def __init__(self, hyper, regressor=None, history=False):
        super().__init__()
  
        self.in_channels_dict = {node_type: len(features) 
                                 for node_type, features in hyper.features.items()}
        
        self.out_channels = hyper.layer_size 
        self.num_layer = hyper.layers
        self.metadata = hyper.metadata  # heterogeneous graph structure
        self.out_win = hyper.out_win
        self.seq_len = hyper.window
        self.device = hyper.device
        self.history = history
        # self.dim = {('grain', 'push', 'joint'):1}
        self.dim = {'joint':2, 'grain':1}
        self.scaling = {'grain':20, 'joint':5}
        
        linear_outchannels = 3*self.out_channels if self.history else 2*self.out_channels
        ## networks
        if regressor:
            self.gclstm_encoder = copy.deepcopy(regressor.gclstm_encoder)
            self.gclstm_decoder = copy.deepcopy(regressor.gclstm_decoder)
         #   self.lin1 = regressor.linear['joint']
            
        else:
            
            self.gclstm_encoder = SeqGCLSTM(self.in_channels_dict, self.out_channels,\
                                            self.num_layer, self.metadata, self.device)
            self.gclstm_decoder = SeqGCLSTM(self.in_channels_dict, self.out_channels, \
                                            self.num_layer, self.metadata, self.device)
            
            
        if self.history:
            self.LSTM = LSTM(self.in_channels_dict, self.out_channels, self.num_layer, self.device,
                             self.dim, seq_len = self.seq_len)
            
         
        self.lin1 = nn.Linear(linear_outchannels+1, 2) # predict length
        self.lin2 = nn.Linear(linear_outchannels+1, 1) # predict probability

        
    def forward(self, x_dict, edge_index_dict, edge_attr):   
    
        if self.history:
            history_encoded = self.LSTM(edge_attr['joint', 'connect', 'joint'])
        
      #  x_dict = {node_type: x_dict[node_type][:, :-(self.seq_len-1)*dim]
      #       for node_type, dim in self.dim.items()}
# edge_attr['joint', 'connect', 'joint'] = edge_attr['joint', 'connect', 'joint'][:, :1]

        hidden_state = self.gclstm_encoder(x_dict, edge_index_dict, edge_attr, None) # all layers of [h, c]
            
        hidden_state = self.gclstm_decoder(x_dict, edge_index_dict, edge_attr, hidden_state)
        
        h_dict, c_dict = hidden_state[-1]
        
        
        """
        
        GrainNN specific topological event
        
        """            
        

        joint_feature = h_dict['joint']
        joint_edge_index = edge_index_dict['joint', 'connect', 'joint']
        
        src, dst = joint_edge_index[0], joint_edge_index[1]

        # concatenate features [h_i, h_j], size (|Ejj|, 2*Dh)

        pair_feature = torch.cat([joint_feature[src], joint_feature[dst], edge_attr['joint', 'connect', 'joint']], dim=-1)
        
        if self.history:
            pair_feature = torch.cat([pair_feature, history_encoded], dim = -1)
           
        y_dict = {'edge_event': self.lin2(pair_feature).view(-1)} # p(i,j), size (Ejj,)

        y_dict['edge'] = torch.tanh(self.lin1(pair_feature)) 

        return y_dict


    def update(self, x_dict, edge_index_dict, edge_attr, y_dict, mask):            
            

        E_pp = edge_index_dict['joint', 'connect', 'joint']
        E_pq = edge_index_dict['joint', 'pull', 'grain']
    #    E_qp = edge_index_dict['grain', 'push', 'joint']
     #   edge_len = edge_attr['joint', 'connect', 'joint'][:,0]



        ''' predict eliminated edge '''
        
        src, dst = E_pp[0], E_pp[1]
        prob = torch.sigmoid(y_dict['edge_event'])
        L1 = ((prob>self.threshold)&(src<dst)).nonzero().view(-1)
        
        
        """
        Grain elimination
        """
        self.elim = {'grain':[], 'joint':[]}
        
        unexpected_elim = []
        
        for grain in y_dict['grain_event']:

           # print(grain)         
   
            Np = E_pq[0][(E_pq[1]==grain).nonzero().view(-1)]
            
            
            if len(Np)==0:
                continue
            
            pairs = torch.combinations(Np, r=2)
            L2 = []
            Nq = []

           # print(pairs)
            for p1, p2 in pairs:
                if p1>p2:
                    p1, p2 = p2, p1
                E_index = ((E_pp[0]==p1)&(E_pp[1]==p2)).nonzero().view(-1)
                if len(E_index)>0:
                    # print(p1, p2)
                    L2.append(E_index)
                    Nq1 = E_pq[1][((E_pq[0]==p1)&(E_pq[1]!=grain)).nonzero().view(-1)]
                    Nq2 = E_pq[1][((E_pq[0]==p2)&(E_pq[1]!=grain)).nonzero().view(-1)]
                    if Nq1[0] in Nq2:
                        Nq.append(Nq1[0])
                    elif Nq1[1] in Nq2:
                        Nq.append(Nq1[1])
                    else: 
                        raise KeyError
                   # print(p1, p2, Nq[-1])
                            
            L2 = torch.cat(L2)
            Nq = torch.tensor(Nq)

           # print('grain', grain, 'eliminate edges', L2)
           # print('grain', int(grain), ', junction neighbors', Np, 'grain neighbors', Nq)
            assert len(Nq)==len(Np) #or len(Np)==2, (grain, Np, Nq)
            
            if len(torch.unique(Nq)) != len(Nq):
                continue
            
            sorted_prob, indices = torch.sort(y_dict['grain'][Nq, 0])
            L2 = L2[indices[:-2]]

            
            force_elim = self.switching_edge_index(E_pp, E_pq, x_dict, y_dict, L2, elim_grain=grain) #x_dict['grain'][grain,:2])
            unexpected_elim.extend(force_elim)
                             
            
           # edge_index_dict['joint', 'connect', 'joint'], edge_index_dict['joint', 'pull', 'grain'] = self.delete_grain_index(grain, E_pp, E_pq, mask)
           # E_pp = edge_index_dict['joint', 'connect', 'joint']
           # E_pq = edge_index_dict['joint', 'pull', 'grain']
            force_elim.insert(0,grain)
            if len(force_elim)>1:
                print('force eliminated grains', force_elim[1:])
                
            for fg in force_elim:
                edge_index_dict['joint', 'connect', 'joint'], edge_index_dict['joint', 'pull', 'grain'] = self.delete_grain_index(fg, E_pp, E_pq, mask)   
                E_pp = edge_index_dict['joint', 'connect', 'joint']
                E_pq = edge_index_dict['joint', 'pull', 'grain']
                    
            for E_index in L2:
               # print(E_index, L1)
                if E_index in L1:
                    L1 = L1[L1!=E_index]
        
        
            ''' ensure there is no two-side grain '''
            
            grains, counts = torch.unique(E_pq[1,:], return_counts=True)
            
            remain_twoside = grains[counts<=2]
            
            for fg in remain_twoside:
                print('find remaining two-side grains', fg)
                edge_index_dict['joint', 'connect', 'joint'], edge_index_dict['joint', 'pull', 'grain'] = self.delete_grain_index(fg, E_pp, E_pq, mask)   
                E_pp = edge_index_dict['joint', 'connect', 'joint']
                E_pq = edge_index_dict['joint', 'pull', 'grain']   
                  
                    
        """
        Neigbor switching
        """

        # print('neigbor switching')
        sorted_prob, indices= torch.sort(prob[L1], dim=0, descending=True)
        L1 = L1[indices]

        for E_index in L1:
             if E_pp[0, E_index] == -1:
                 L1 = L1[L1!=E_index]

       # print('edge switching index', L1)
       # print(E_pp.T[L1])

        force_elim = self.switching_edge_index(E_pp, E_pq, x_dict, y_dict, L1, elim_grain=None)
        #    assert len(force_elim)==0
        
        switching_list = E_pp.T[L1]
        
        ''' ensure there is no two-side grain '''
        
        grains, counts = torch.unique(E_pq[1,:], return_counts=True)
        
        remain_twoside = grains[counts<=2]
        
        for fg in remain_twoside:
            print('find remaining two-side grains', fg)
            edge_index_dict['joint', 'connect', 'joint'], edge_index_dict['joint', 'pull', 'grain'] = self.delete_grain_index(fg, E_pp, E_pq, mask)   
            E_pp = edge_index_dict['joint', 'connect', 'joint']
            E_pq = edge_index_dict['joint', 'pull', 'grain']   
            
        unexpected_elim.extend(remain_twoside)    
        if len(unexpected_elim)>0:
            unexpected_elim = torch.tensor(unexpected_elim)
            y_dict['grain_event'] = torch.cat([y_dict['grain_event'], unexpected_elim])    
       # assert torch.all(counts>2)
       # print(switching_list)
        
        edge_index_dict['joint', 'connect', 'joint'], edge_index_dict['joint', 'pull', 'grain'] = self.cleanup(E_pp, E_pq)
        
        edge_index_dict['grain', 'push', 'joint'] = torch.flip(edge_index_dict['joint', 'pull', 'grain'], dims=[0])
      
        
        
                
        return edge_index_dict, switching_list


    def cleanup(self, E_pp, E_pq):
        """
        for grain in self.elim['grain']:
            E_pq = E_pq[:, (E_pq[1]!=grain).nonzero().view(-1)]
        
        for joint in self.elim['joint']:
            E_pq = E_pq[:, (E_pq[0]!=joint).nonzero().view(-1)]
            E_pp = E_pp[:, (E_pp[0]!=joint).nonzero().view(-1)]
            E_pp = E_pp[:, (E_pp[1]!=joint).nonzero().view(-1)]
            
        """
        
        E_pq = E_pq[:, (E_pq[0]!=-1).nonzero().view(-1)]
        E_pp = E_pp[:, (E_pp[0]!=-1).nonzero().view(-1)]
        
        
        return E_pp, E_pq
        
    def delete_grain_index(self, grain, E_pp, E_pq, mask):
        
        Np = E_pq[0][(E_pq[1]==grain).nonzero().view(-1)]
        assert len(Np) == 2, Np
        
        ''' find the two to delete p1, p2'''
        p1, p2 = Np
        #print('remained vertices to delete', p1, p2)
        ''' find the two to connect p1_c, p2_c'''
        Np1 = E_pp[1][((E_pp[0]==p1)&(E_pp[1]!=p2)).nonzero().view(-1)][0]
        Np2 = E_pp[1][((E_pp[0]==p2)&(E_pp[1]!=p1)).nonzero().view(-1)][0]
        
       # print('new connection', Np1, Np2)
        ''' remove all the edges connected to p1, p2'''
        
        ''' add one edge between p1_c, p2_c'''
        E_pp = torch.cat([E_pp, torch.tensor([[Np1, Np2],[Np2, Np1]])], dim=-1)
        
        mask['grain'][grain] = 0
        mask['joint'][p1] = 0
        mask['joint'][p2] = 0
        
        E_pq[:, (E_pq[1]==grain).nonzero().view(-1)] = -1
        
        for joint in [p1, p2]:
            E_pq[:, (E_pq[0]==joint).nonzero().view(-1)] = -1
            E_pp[:, (E_pp[0]==joint).nonzero().view(-1)] = -1
            E_pp[:, (E_pp[1]==joint).nonzero().view(-1)] = -1
       # self.elim['grain'].append(grain)
       # self.elim['joint'].append(p1)
       # self.elim['joint'].append(p2)
        
        return E_pp, E_pq
    

    def switching_edge_index(self, E_pp, E_pq, x_dict, y_dict, elimed_arg, elim_grain):

        force_elim = []
        pairs = torch.unique(E_pp.T[elimed_arg].view(-1))
        save_prev = {}
        for p in pairs:
            x_dict['joint'][p, :2] -= y_dict['joint'][p]/self.scaling['joint'] 
            save_prev[int(p)] = x_dict['joint'][p, :2]
        
        for index in range(len(elimed_arg)):
            p1, p2 = E_pp.T[elimed_arg][index]

           # print(E_pp.T[elimed_arg])
           # print(p1, p2)
            
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
                        
            # find two expanding grains and two shrinking grains
            expand_q1 = p1_qn[(1-sum(p1_qn==i for i in p2_qn)).nonzero(as_tuple=True)] # new neighbor for p2
            expand_q2 = p2_qn[(1-sum(p2_qn==i for i in p1_qn)).nonzero(as_tuple=True)] # new neighbor for p1
           # print(p1_qn, p2_qn)
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
            

                
            if elim_grain is None and (sq1_p1==sq1_p2 or sq2_p1==sq2_p2):
               # print('edge cannot affect grain', sq1_p1, sq1_p2, sq2_p1,sq2_p2)
                continue  
            
            if sq1_p1==sq1_p2:
                if shrink_q1!=elim_grain:
                    force_elim.append(shrink_q1)
                
            if sq2_p1==sq2_p2:
                if shrink_q2!=elim_grain:
                    force_elim.append(shrink_q2)            
           # print('\n',p1_pn, p2_pn)
          #  print(p1_pn_index, p2_pn_index)
            
            ''' coordinates of two new vertices '''
            
            x_p1 = x_dict['joint'][p1, :2] 
            x_p2 = x_dict['joint'][p2, :2]
            
            x_p2_p = periodic_move(x_p2, x_p1)
            c_p1p2 = 0.5*( x_p1 + x_p2_p )              
            x_dict['joint'][p1, :2], x_dict['joint'][p2, :2] = c_p1p2, periodic_move(c_p1p2, x_p2)  
            
            
          #  x_dict['joint'][p1, :2], x_dict['joint'][p2, :2] = rotate_two_points(x_p1, x_p2)

            
            
            swap = False
            if point_in_triangle(x_dict['joint'][p2,:2], x_dict['joint'][p1,:2], \
                                 x_dict['joint'][sq1_p1,:2], x_dict['joint'][sq1_p2,:2]): 
                swap = True
         #   print(pairs.view(-1))  
         #   print(sq1_p1, sq1_p2, sq2_p1, sq2_p2)
            nxt = torch.unique(E_pp.T[elimed_arg][index:].view(-1))
            if sq1_p2 in nxt and sq2_p2 not in nxt:
                swap = False
            if sq2_p2 in nxt and sq1_p2 not in nxt:
                swap = True               
            if sq1_p1 in nxt and sq2_p1 not in nxt:
                swap = True
            if sq2_p1 in nxt and sq1_p1 not in nxt:
                swap = False            
         #   print(swap)
                
            if swap:
               # print('swap')
                # swap
                p1_qn_index_sort.reverse()
                p2_qn_index_sort.reverse()
                p1_pn_index.reverse()
                p2_pn_index.reverse()
                
                sq1_p1, sq2_p1 = sq2_p1, sq1_p1
                sq1_p2, sq2_p2 = sq2_p2, sq1_p2
            

            
              
            # replace joint-grain edges

            E_pq[1, p1_qn_index_sort[1]] = expand_q2 # reject sq2
            E_pq[1, p2_qn_index_sort[0]] = expand_q1
            
     
            # replace joint-joint edges
           # print((sq1_p2, p2),'->', (sq1_p2, p1))
           # print((sq2_p1, p1),'->', (sq2_p1, p2))   
            
            E_pp[0, p1_pn_index[1]] = p2
            E_pp[0, p2_pn_index[0]] = p1
                                
            E_pp[1][ ((E_pp[0]==sq1_p2)&(E_pp[1]==p2)).nonzero().view(-1) ] = p1
            E_pp[1][ ((E_pp[0]==sq2_p1)&(E_pp[1]==p1)).nonzero().view(-1) ] = p2
      
       # print(E_pp.T[elimed_arg])
       # print(save_prev)
        for p in pairs:
            y_dict['joint'][p] = self.scaling['joint']*(x_dict['joint'][p,:2] - save_prev[int(p)])           
            x_dict['joint'][p, 6:8] =  y_dict['joint'][p]  
        

        return force_elim

def point_in_triangle(t, v1, v2, v3):
    sign = lambda a, b, c: (a[0] - c[0])*(b[1] - c[1]) - \
                           (b[0] - c[0])*(a[1] - c[1])
   # print(v1, t)
    v1_m = periodic_move(v1, t)
    v2_m = periodic_move(v2, t)
    v3_m = periodic_move(v3, t)
  #  print(t, v1, v2, v3)
    d1 = sign(t, v1_m, v2_m)
    d2 = sign(t, v2_m, v3_m)
    d3 = sign(t, v3_m, v1_m)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg & has_pos)

def rotate_two_points(x_p1_p, x_p2_p):
    
    x_p2_p_m = periodic_move(x_p2_p, x_p1_p)
    
    c_p1p2 = 0.5*( x_p1_p + x_p2_p_m )
    x_p1 = x_p1_p - c_p1p2
    x_p2 = x_p2_p_m - c_p1p2
    
    x_p1[0], x_p1[1] = 0, 0 #-x_p1[1], x_p1[0]
    x_p2[0], x_p2[1] = 0, 0 #-x_p2[1], x_p2[0]
    x_p1 += c_p1p2
    x_p2 += c_p1p2
    
    x_p2 = periodic_move(x_p2, x_p2_p)
    
    return x_p1, x_p2


'''
def periodic_move(p, pc):
    x,  y  = p
    xc, yc = pc

    rel_x = x - xc
    rel_y = y - yc
    a = x -1*(rel_x>0.5) + 1*(rel_x<-0.5) 
    b = y -1*(rel_y>0.5) + 1*(rel_y<-0.5) 

    return torch.tensor([a, b])

'''
def periodic_move(p, pc):
    rel_x = p - pc

    return p -1*(rel_x>0.5) + 1*(rel_x<-0.5) 


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
"""
<1> approximate with pure rotation
<2> improve
p1_p2 = ((E_pp[0]==p1)&(E_pp[1]==p2)).nonzero().view(-1)
p2_p1 = ((E_pp[0]==p2)&(E_pp[1]==p1)).nonzero().view(-1)

x_dict['joint'][p1, :2]  = x_dict['joint'][p1, :2] - y_dict['joint'][p1] + y_dict['edge_rotation'][p1_p2]
x_dict['joint'][p2, :2]  = x_dict['joint'][p2, :2] - y_dict['joint'][p2] + y_dict['edge_rotation'][p2_p1]

x_dict['joint'][p1, -2:] =  y_dict['edge_rotation'][p1_p2]
x_dict['joint'][p2, -2:] =  y_dict['edge_rotation'][p2_p1]


           # print(grain_sort_index)
           # print(prob[L2])
           # L2 = L2[grain_sort_index[:-2]]
            
           # sorted_prob, indices = torch.sort(prob[L2], dim=0, descending=True)
           # print('orientation', x_dict['grain'][Nq, 7])
           # print('prob', prob[L2])
            
           # sorted_prob, indices= torch.sort((x_dict['grain'][Nq, 7]-sq2)**2, dim=0)
           # sorted_prob, indices= torch.sort(edge_len[L2], dim=0)
"""