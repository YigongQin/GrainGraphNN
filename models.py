#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 11:34:53 2021

@author: yigongqin
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor
import torch.nn.init as init
from typing import Callable, List, Optional, Tuple
from torch_geometric_temporal.nn.hetero import HeteroGCLSTM


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, device):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        input shape: N, (2+num_param), G
        hidden sshape: N, hidden_dim, G
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = (kernel_size[0]-1) // 2
        self.bias = bias
        self.device = device
        self.weight_ci = Parameter(torch.empty((self.hidden_dim, self.hidden_dim), dtype = torch.float64, device = device))
        self.weight_cf = Parameter(torch.empty((self.hidden_dim, self.hidden_dim), dtype = torch.float64, device = device))
        self.weight_co = Parameter(torch.empty((self.hidden_dim, self.hidden_dim), dtype = torch.float64, device = device))
        self.reset_parameters()

        ''' 
        self.conv = nn.Conv1d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        '''
        self.conv = self_attention(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              bias=self.bias,
                              device=self.device)


    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input_tensor, active, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined, active)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)


        sc_i = torch.einsum('biw, oi -> bow', c_cur, self.weight_ci) 
        sc_f = torch.einsum('biw, oi -> bow', c_cur, self.weight_cf) 

        i = torch.sigmoid(cc_i + sc_i)
        f = torch.sigmoid(cc_f + sc_f)
        c_next = f * c_cur + i * torch.tanh(cc_g)

        sc_o = torch.einsum('biw, oi -> bow', c_next, self.weight_co) 

        o = torch.sigmoid(cc_o + sc_o)
        
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, width, dtype=torch.float64, device=self.device),
                torch.zeros(batch_size, self.hidden_dim, width, dtype=torch.float64, device=self.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, W or T, B, C, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 device, batch_first=True, bias=True, return_all_layers=True):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.device = device

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                          device=self.device))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state):
        '''
        input for ConvLSTM B, T, C, W 

        '''
       # if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
       #     input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
       
        b, seq_len, channel, w = input_tensor.size()

        # Implement stateful ConvLSTM
       # if hidden_state is not None:
       #     raise NotImplementedError()
       # else:
        if hidden_state is None:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=w)

        layer_output_list = []
        last_state_list = []

         
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :], 
                                                 cur_state=[h, c])
                ## output shape b, hidden_dim, w
                output_inner.append(h)
                
            ##stack every time step to form output, shape is b, t, hidden, w
            layer_output = torch.stack(output_inner, dim=1) 
            
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c]) ## 

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        ## init the hidden states for every layer
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
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
    
    
    
    
class ConvLSTM_seq(nn.Module):
    def __init__(self, hyper, device):
        super(ConvLSTM_seq, self).__init__()
  
        self.input_dim = hyper.feature_dim  ## this input channel
        self.hidden_dim = hyper.layer_size ## this output_channel
        self.num_layer = hyper.layers
        self.w = hyper.G_base
        self.out_win = hyper.out_win
        self.kernel_size = hyper.kernel_size
        self.bias = hyper.bias
        self.device = device
        self.dt = hyper.dt

        ## networks
        self.lstm_encoder = ConvLSTM(self.input_dim, self.hidden_dim, self.kernel_size, self.num_layer[0], self.device)
        self.lstm_decoder = ConvLSTM(self.input_dim, self.hidden_dim, self.kernel_size, self.num_layer[1], self.device)

        self.project = nn.Linear(self.hidden_dim*self.w, self.w)## make the output channel 1
        self.project_y = nn.Linear(self.hidden_dim*self.w, 1)
        self.project_a = nn.Linear(self.hidden_dim*self.w, self.w)

    def forward(self, input_seq, Cl):
        

        ## step 1 remap the input to the channel with gridDdim G
        ## b,t, input_len -> b,t,c,w 
        b, t, c, w  = input_seq.size()
        
        output_seq = torch.zeros(b, self.out_win, 2*self.w+1, dtype=torch.float64).to(self.device)
        frac_seq = torch.zeros(b, self.out_win, self.w,   dtype=torch.float64).to(self.device)

        seq_1 = input_seq[:,-1,:,:]    # the last frame

        encode_out, hidden_state = self.lstm_encoder(input_seq, None)  # output range [-1,1], None means stateless LSTM
        
        
        for i in range(self.out_win):
            
            encode_out, hidden_state = self.lstm_decoder(seq_1.unsqueeze(dim=1),hidden_state)
            last_time = encode_out[-1][:,-1,:,:].view(b, self.hidden_dim*self.w)
            
            dy = F.relu(self.project_y(last_time))    # [b,1]
            darea = (self.project_a(last_time))    # [b,1]
            dfrac = self.project(last_time)/Cl   # project last time output b,hidden_dim, to the desired shape [b,w]   
            frac = F.relu(dfrac+seq_1[:,0,:])         # frac_ini here is necessary to keep
            frac = F.normalize(frac, p=1, dim=-1)  # [b,w] normalize the fractions
            
            dfrac = (frac - seq_1[:,0,:])/frac_norm 
 
            
            output_seq[:,i, :self.w] = dfrac
            output_seq[:,i, self.w:2*self.w] = F.relu(darea)
            output_seq[:,i, -1:] = dy
            frac_seq[:,i,:] = frac
            ## assemble with new time-dependent variables for time t+dt: FRAC, Y, T  [b,c,w]
            
            seq_1 = torch.cat([frac.unsqueeze(dim=1), dfrac.unsqueeze(dim=1), darea.unsqueeze(dim=1), \
                    dy.expand(-1,self.w).view(b,1,self.w), seq_1[:,4:-1,:], seq_1[:,-1:,:] + self.dt ],dim=1)

                        
        return output_seq, frac_seq




class ConvLSTM_start(nn.Module):
    def __init__(self, hyper,device):
        super(ConvLSTM_start, self).__init__()
        self.input_dim = hyper.feature_dim  ## this input channel
        self.hidden_dim = hyper.layer_size ## this output_channel
        self.num_layer = hyper.layers
        self.w = hyper.G_base
        self.out_win = hyper.out_win - 1
        self.kernel_size = hyper.kernel_size
        self.bias = hyper.bias
        self.device = device
        self.dt = hyper.dt

        ## networks
        self.lstm_decoder = ConvLSTM(self.input_dim, self.hidden_dim, self.kernel_size, self.num_layer[1], self.device)
        self.project = nn.Linear(self.hidden_dim*self.w, self.w)## make the output channel 1
        self.project_y = nn.Linear(self.hidden_dim*self.w, 1)
        self.project_a = nn.Linear(self.hidden_dim*self.w, self.w)


        
    def forward(self, input_seq, Cl):
        

        ## step 1 remap the input to the channel with gridDdim G
        ## b,t, input_len -> b,t,c,w 
        b, t, c, w = input_seq.size()
        
        output_seq = torch.zeros(b, self.out_win, 2*self.w+1, dtype=torch.float64).to(self.device)
        frac_seq = torch.zeros(b, self.out_win, self.w,   dtype=torch.float64).to(self.device)

        seq_1 = input_seq[:,-1,:,:]    # the last frame
        

        
        
        for i in range(self.out_win):
            
            encode_out, hidden_state = self.lstm_decoder(seq_1.unsqueeze(dim=1), None)
            last_time = encode_out[-1][:,-1,:,:].view(b, self.hidden_dim*self.w)
            
            dy = F.relu(self.project_y(last_time))    # [b,1]
            darea = (self.project_a(last_time))    # [b,1]
            dfrac = self.project(last_time)/Cl   # project last time output b,hidden_dim, to the desired shape [b,w]   
            frac = F.relu(dfrac+seq_1[:,0,:])         # frac_ini here is necessary to keep
            frac = F.normalize(frac, p=1, dim=-1)  # [b,w] normalize the fractions
            
            dfrac = (frac - seq_1[:,0,:])/frac_norm 
 
            output_seq[:,i, :self.w] = dfrac
            output_seq[:,i, self.w:2*self.w] = F.relu(darea)
            output_seq[:,i, -1:] = dy
            frac_seq[:,i,:] = frac
            ## assemble with new time-dependent variables for time t+dt: FRAC, Y, T  [b,c,w]
            
            seq_1 = torch.cat([frac.unsqueeze(dim=1), dfrac.unsqueeze(dim=1), darea.unsqueeze(dim=1), \
                    dy.expand(-1,self.w).view(b,1,self.w), seq_1[:,4:-1,:], seq_1[:,-1:,:] + self.dt ],dim=1)

                        
        return output_seq, frac_seq









