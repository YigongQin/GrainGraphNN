import torch
import numpy as np



def assemb_seq(frac, dfrac, area, y):

    if frac.ndim==3:
       return np.concatenate((frac, dfrac, area, y[:,:,np.newaxis]), axis=-1)
    if frac.ndim==2: ## batch size = 1
       return np.concatenate((frac, dfrac, area, y[:,np.newaxis]), axis=-1)

def divide_seq(seq, G):

    if seq.ndim==3:
       return seq[:,:,:G], seq[:,:,G:2*G], seq[:,:,2*G:3*G], seq[:,:,-1]
    if seq.ndim==2:
       return seq[:,:G], seq[:,G:2*G], seq[:,2*G:3*G], seq[:,-1]


def assemb_feat(loss_feature, frac, G, output):

    output[:,:,0,:] = frac
    output[:,:,1,:] = loss_feature[:,:,:G]
    output[:,:,2,:] = loss_feature[:,:,G:2*G]
    output[:,:,3,:] = loss_feature[:,:,-1:]

def divide_feat(seq):
    if seq.ndim==4:
       return seq[:,:,0,:], seq[:,:,1,:], seq[:,:,2,:], seq[:,:,3,0]
    if seq.ndim==3:
       return seq[:,0,:], seq[:,1,:], seq[:,2,:], seq[:,3,0]