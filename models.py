#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 11:34:53 2021

@author: yigongqin
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from heteropgclstm import HeteroPGCLSTM
from heterogclstm import HeteroGCLSTM



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
    
    
    
    
class GrainNN2(nn.Module):
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

        self.linear = nn.ModuleDict({node_type: nn.Linear(self.out_channels, len(targets))
                        for node_type, targets in hyper.targets.items()}) 

        self.jj_edge = nn.Linear(2*self.out_channels, 1)

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
        
        
        
        for i in range(self.out_win):
            
            hidden_state = self.gclstm_decoder(x_dict, edge_index_dict, hidden_state)
            h_dict, c_dict = hidden_state[-1]
            
            y_dict = {node_type: self.linear[node_type](h)
                 for node_type, h in h_dict.items()} # apply the linear layer
            
            """
            
            GrainNN specific y output 
            
            """
            
            y_dict['joint'] = torch.sigmoid(y_dict['joint']) - 0.5 # dx, dy are in the range [-1, 1]
            
            area = F.relu(y_dict['grain'][:, 0] + x_dict['grain'][:, 3]) # darea + area_old is positive    
            
            area = F.normalize(area, p=1, dim=-1)  # normalize the area
            
            y_dict['grain'][:, 0] = area - x_dict['grain'][:, 3]
 
            y_dict['grain'][:, 1] = F.relu(y_dict['grain'][:, 1]) # excess volume predict
            
            
            """
            
            GrainNN specific topological event
            
            """            
            
            event_dict = {}
            event_dict.update({'grain': 1*(area<1e-6) })
            event_dict.update({'edge': 1})
            
            """
            
            Online update for next snapshot input x
            
            """    
            
            x_dict = self.form_next_x(x_dict, y_dict)
            edge_index_dict = self.from_next_edge_index(edge_index_dict, event_dict)
            
                        
        return y_dict
    
    @staticmethod
    def form_next_x(x_dict, y_dict):
        
        # features
        x_dict['joint'][:, :2]  += y_dict['joint']
        x_dict['grain'][:, 3]   += y_dict['grain'][:, 0]
        x_dict['grain'][:, 4]   =  y_dict['grain'][:, 1]
        
        # gradients
        x_dict['joint'][:, -2:] =  y_dict['joint']
        x_dict['grain'][:, -1]  =  y_dict['grain'][:, 0]
        
        return x_dict

    @staticmethod
    def from_next_edge_index(edge_index_dict, event_dict):
        
        return edge_index_dict
        


"""

  self.features = {'grain':['x', 'y', 'z', 'area', 'extraV', 'cosx', 'sinx', 'cosz', 'sinz'],
                   'joint':['x', 'y', 'z', 'G', 'R']}
#  self.features_edge = ['len']
  
  
  self.features_grad = {'grain':['darea'], 'joint':['dx', 'dy']}
  
  
  self.targets = {'grain':['darea', 'extraV'], 'joint':['dx', 'dy']}    

"""
