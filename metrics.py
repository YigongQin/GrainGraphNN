#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 13:43:45 2022

@author: yigongqin
"""
import torch
from collections import defaultdict
import math


class feature_metric:
    def __init__(self, model_type, model_id):
        self.model_type = model_type
        self.model_id = model_id
        self.metric_list = []
        
        self.acc_dicts = defaultdict(float)
        self.class_dicts = defaultdict(list)
        
    def record(self, y_dict, pred, mask, epoch):

        def add(key, idx):
           # print((( mask[key][:,0]*(data[key][:,idx] - pred[key][:,idx])**2 )))
           # print((( mask[key][:,0]*(data[key][:,idx])**2 )))
            self.acc_dicts[key+str(idx)+'err'] += float(torch.sum( mask[key][:,0]*(y_dict[key][:,idx] - pred[key][:,idx])**2 ))
            if epoch == 0:
                self.acc_dicts[key+str(idx)] += float(torch.sum( mask[key][:,0] *(y_dict[key][:,idx])**2 ))
        
        def PR(key, threshold, y, prob):
            
            for i, t in enumerate(threshold):
                self.acc_dicts['TP'][i] += float(sum( (y==1) & (prob>t) )) 
                self.acc_dicts['FP'][i] += float(sum( (y==0) & (prob>t) ))
                self.acc_dicts['FN'][i] += float(sum( (y==1) & (prob<=t) ))
                
        if self.model_type == 'regressor':
            
            add('grain', 0)
            add('grain', 1)
            add('joint', 0)
            add('joint', 1)        

        

        if self.model_type == 'classifier':
            
            pass
            
    
    def epoch_summary(self):
        
        if self.model_type == 'classifier':
         #   train_auc, P_list, R_list = class_acc(train_prob, train_label)
            self.test_auc, self.P_list, self.R_list = class_acc(self.test_prob, self.test_label)
            print('Validation AUC:{:.6f}'.format(self.test_auc)) 
            self.metric_list.append(float(self.test_auc))
         #   print('Train AUC:{:.6f}, valid AUC:{:.6f}'.format(train_auc, test_auc)) 

        if self.model_type == 'regressor':

            x_err = 100*math.sqrt(self.acc_dicts['joint0err']/self.acc_dicts['joint0'])
            y_err = 100*math.sqrt(self.acc_dicts['joint1err']/self.acc_dicts['joint1'])
            s_err = 100*math.sqrt(self.acc_dicts['grain0err']/self.acc_dicts['grain0'])
            v_err = 100*math.sqrt(self.acc_dicts['grain1err']/self.acc_dicts['grain1'])
            
            print('err, joint x: %2.1f, y: %2.1f, grain s: %2.1f, v: %2.1f'%(x_err, y_err, s_err, v_err))
            self.acc_dicts['joint0err'] = 0
            self.acc_dicts['joint1err'] = 0
            self.acc_dicts['grain0err'] = 0
            self.acc_dicts['grain1err'] = 0            
            

    def summary(self):


        if self.model_type == 'classifier':
            print('model id:', self.model_id, 'PR AUC', float(self.test_auc))
            self.plist = [float(i) for i in self.P_list]
            self.rlist = [float(i) for i in self.R_list]
        
        if self.model_type == 'regressor':
            print('model id:', self.model_id, 'ACCURACY', self.acc_dicts)



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