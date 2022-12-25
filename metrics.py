#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 13:43:45 2022

@author: yigongqin
"""
import torch



def regress_acc(data, pred, mask, acc_dicts, epoch):

    def add(key, idx):
       # print((( mask[key][:,0]*(data[key][:,idx] - pred[key][:,idx])**2 )))
       # print((( mask[key][:,0]*(data[key][:,idx])**2 )))
        acc_dicts[key+str(idx)+'err'] += float(torch.sum( mask[key][:,0]*(data[key][:,idx] - pred[key][:,idx])**2 ))
        if epoch == 0:
            acc_dicts[key+str(idx)] += float(torch.sum( mask[key][:,0] *(data[key][:,idx])**2 ))
       # print(mask)

    add('grain', 0)
    add('grain', 1)
    add('joint', 0)
    add('joint', 1)

    

    """
    # use PRAUC
    
    area = torch.sigmoid(torch.cat(prob))
    y = torch.cat(label)

    AUC = 0
    intervals = 5
    P_list, R_list = [], []
    left_bound = 0

    for i in range(intervals+1): 
        # the first one is all positive, no negative, recall is one
        threshold = 1 - i/intervals

        TruePositive  = sum( (y==1) & (prob>threshold) )
        FalsePositive = sum( (y==0) & (prob>threshold) )
        FalseNegative = sum( (y==1) & (prob<=threshold) )
        
        
        if TruePositive + FalsePositive>0 and TruePositive + FalseNegative>0:
            
            # it's a valid data
            
            Precision = TruePositive/(TruePositive + FalsePositive) 
            Recall = TruePositive/(TruePositive + FalseNegative)
    
            AUC += (Recall-left_bound)*Precision
            left_bound = Recall
        
        else:
            Precision = -1
            Recall = -1
        
        P_list.append(Precision)
        R_list.append(Recall)
    
    return AUC, P_list, R_list
    """


def class_acc(prob, label):
    
    # use PRAUC
    
    prob = torch.sigmoid(torch.cat(prob))
    y = torch.cat(label)

    AUC = 0
    intervals = 10
    P_list, R_list = [], []
    left_bound = 0

    for i in range(intervals+1): 
        # the first one is all positive, no negative, recall is one
        threshold = 1 - i/intervals

       # print(threshold)

        TruePositive  = sum( (y==1) & (prob>threshold) )
        FalsePositive = sum( (y==0) & (prob>threshold) )
        FalseNegative = sum( (y==1) & (prob<=threshold) )
        
        
        if TruePositive + FalsePositive>0 and TruePositive + FalseNegative>0:
            
            # it's a valid data
            
            Precision = TruePositive/(TruePositive + FalsePositive) 
            Recall = TruePositive/(TruePositive + FalseNegative)
    
            AUC += (Recall-left_bound)*Precision
            left_bound = Recall
        
        else:
            Precision = -1
            Recall = -1
        
        P_list.append(Precision)
        R_list.append(Recall)
       # print(Precision, Recall, left_bound)
  # print(Positive, TruePositive, FalsePositive)
  #  print(Presicion, Recall, F1)
   # return F1 if Positive else -1
    return AUC, P_list, R_list


         
def edge_error_metric(data_edge_index, pred_edge_index):
    
    unorder_edge = lambda a:set(map(tuple, a))

    E_pp = unorder_edge(data_edge_index['joint', 'connect', 'joint'].detach().numpy().T)
    E_pq = unorder_edge(data_edge_index['joint', 'pull', 'grain'].detach().numpy().T)
    
    E_t_pp = unorder_edge(pred_edge_index['joint', 'connect', 'joint'].detach().numpy().T)
    E_t_pq = unorder_edge(pred_edge_index['joint', 'pull', 'grain'].detach().numpy().T) 

    return 1-len(E_pp.intersection(E_t_pp))/len(E_pp), \
           1-len(E_pq.intersection(E_t_pq))/len(E_pq)