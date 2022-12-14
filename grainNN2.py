#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:35:27 2022

@author: yigongqin
"""
import argparse, time, glob, dill, random, os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel
from data_loader import DynamicHeteroGraphTemporalSignal
from models import GrainNN_classifier, GrainNN_regressor, regressor_classifier, rot90
from parameters import regressor, classifier, classifier_transfered
from graph_trajectory import graph_trajectory
from metrics import feature_metric, edge_error_metric
from QoI import data_analysis

def criterion(data, pred, mask):
   # print(torch.log(pred['edge_event']))
   # print(data['edge_event'])
    

   # print(p)
    if args.model_type== 'regressor':
        #print(data['grain'][:,:1], pred['grain'][:,:1]) 
      #  return 1000*torch.mean(mask['joint']*(data['joint'] - pred['joint'])**2) \
      #       + 100 *torch.mean(mask['grain']*(data['grain'] - pred['grain'])**2)

        loss = torch.mean(mask['joint']*(data['joint'] - pred['joint'])**2) \
             + torch.mean(mask['grain']*(data['grain'] - pred['grain'])**2)
        return 10*loss

    if args.model_type== 'classifier':
        z = pred['edge_event']
        y = data['edge_event']
        
        qualified_y = torch.where(y>-1)
        y = y[qualified_y]
        z = z[qualified_y]
        
        classifier = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hp.weight))
        
        return classifier(z, y.type(torch.FloatTensor))
    


def train(model, train_loader, test_loader):
    
    if device=='cuda':
        print('use %d GPUs'%torch.cuda.device_count())
        model.cuda()
        
    

    optimizer = torch.optim.Adam(model.parameters(),lr=hp.lr) 
    if args.transfer and args.model_type== 'classifier':
        optimizer = torch.optim.Adam(
        [
            {"params": model.gclstm_encoder.parameters(), "lr": hp.lr_1*hp.lr_2*hp.lr},
            {"params": model.gclstm_decoder.parameters(), "lr": hp.lr_2*hp.lr},
            {"params": model.lin2.parameters()},
        ],
        lr=hp.lr,
        )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=hp.decay_step, gamma=0.5, last_epoch=-1)
  #  torch.autograd.set_detect_anomaly(True)
    metric = feature_metric(args.model_type, args.model_id)

    train_loss, count = 0, 0

    for data in train_loader:  
        train.x = data.x_dict
        train.edge = data.edge_index_dict
        train.y = data.y_dict
        count += 1 #data.batch
        data.to(device)
        pred = model(data.x_dict, data.edge_index_dict)
        train_loss += float(criterion(data.y_dict, pred, data['mask'])) 
    train_loss/=count

    test_loss, count = 0, 0
  #  test_acc_dict = defaultdict(float)
    for data in test_loader:      
        count += 1
        data.to(device)
        pred = model(data.x_dict, data.edge_index_dict)
        test_loss += float(criterion(data.y_dict, pred, data['mask']))  
        
        metric.record(data.y_dict, pred, data['mask'], 0)
   # if args.model_type=='regressor':
   #     regress_acc(data.y_dict, pred, data['mask'], test_acc_dict, 0)
   # print(test_acc_dict)
    test_loss/=count

    print('Epoch:{}, Train loss:{:.6f}, valid loss:{:.6f}'.format(0, float(train_loss), float(test_loss)))
    train_loss_list.append(float(train_loss))
    test_loss_list.append(float(test_loss))  
    train0, test0 = train_loss_list[0], test_loss_list[0]
    
    
    metric.epoch_summary()


    for epoch in range(1, hp.epoch+1):


       # if mode=='train' and epoch==num_epochs-10: optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

        model.train()
        train_loss, count = 0, 0
        for data in train_loader:   
            data.to(device)
            count += 1
            
            pred = model(data.x_dict, data.edge_index_dict)
         
            loss = criterion(data.y_dict, pred, data['mask'])
             
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += float(loss)

        train_loss/=count
        
        model.eval()
        test_loss, count = 0, 0
      #  test_prob, test_label = [], []
      #  test_acc_dict = defaultdict(float)
        with torch.no_grad():
            for data in test_loader:  
            
                count += 1
                data.to(device)
                pred = model(data.x_dict, data.edge_index_dict)
                test_loss += float(criterion(data.y_dict, pred, data['mask'])) 
                metric.record(data.y_dict, pred, data['mask'], epoch)
                
              #  if args.model_type== 'regressor': 
              #      regress_acc(data.y_dict, pred, data['mask'], test_acc_dict, epoch)
                    
                
              #  if args.model_type== 'classifier':       
                     
              #      test_prob.append(pred['edge_event'])
              #      test_label.append(data.y_dict['edge_event'])
                    
     #   if epoch == hp.epoch: print(test_acc_dict)    
        
        test_loss/=count
        
        print('\n')
        print('Epoch:{}, Train loss:{:.6f}, valid loss:{:.6f}'.format(epoch, float(train_loss)/train0, float(test_loss)/test0 ))
        train_loss_list.append(float(train_loss))
        test_loss_list.append(float(test_loss))    
        metric.epoch_summary()
        
        
      #  test_acc_list.append(test_acc_dict)
        
        scheduler.step()
        
    print('model id:', args.model_id, 'loss', test_loss)
    metric.summary()
    
    
    train.metric = metric
        
    
    return model 




if __name__=='__main__':
    

    
    parser = argparse.ArgumentParser("Train the model.")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model_id", type=int, default=0)
    parser.add_argument("--model_exist", type=bool, default=False)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--model_dir", type=str, default='./GR/')
    parser.add_argument("--model_type", type=str, default='regressor')
    parser.add_argument("--data_dir", type=str, default='./sameGR/level2/')
    parser.add_argument("--test_dir", type=str, default='./test/')
    parser.add_argument("--use_sample", type=str, default='all')
    
    parser.add_argument("--plot_flag", type=bool, default=False)
    parser.add_argument("--noPDE", type=bool, default=True)
    parser.add_argument("--transfer", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--symmetry", type=int, default=1)
    
    parser.add_argument("--models", type=tuple, default=(0, 0))
    args = parser.parse_args()
    
    
    mode = args.mode
    model_id = args.model_id
    device = args.device
    
    if mode == 'test': args.model_exist = True
    
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print('torch seed', seed)

    

    
    print('==========  GrainNN specification  =========')
    print('3D grain microstructure evolution')
    print('the mode is: ', mode, ', the model id is: ', model_id)
    print('device: ', args.device)
    print('model already exists, no training required: ', args.model_exist)
    print('no PDE solver required, input is random: ', args.noPDE)
    print('plot GrainNN verus PDE pointwise error: ', args.plot_flag)
    print('\n')
    

    print('************ setup data ***********')
    num_train, num_valid, num_test = 0, 0, 0 
    
    
    if mode == 'train':
    
        datasets = sorted(glob.glob(args.data_dir + 'case*'))
        
        data_list = []
        
        for case in datasets:
            with open(case, 'rb') as inp:  
                try:
                    data_list = data_list + dill.load(inp)
                except:
                    raise EOFError

        num_train = int(args.train_ratio*len(data_list))
        num_valid = len(data_list) - num_train
        sample = data_list[0]
        
    if mode == 'test':
        
        datasets = sorted(glob.glob(args.test_dir + 'case*'))
        
        data_list = []
        
        for case in datasets:
            with open(case, 'rb') as inp:  
                try:
                    data_list = data_list + dill.load(inp)
                except:
                    raise EOFError
                
        num_test = len(data_list)
        sample = data_list[0]
        
  #  random.shuffle(data_list)

    
    train_list = data_list[:num_train]
    if args.use_sample !='all':
        train_list = train_list[:int(args.use_sample)]
    
    test_list = data_list[num_train:]       

    #data_prop = data_analysis(train_list)          
    
    if args.model_type== 'regressor':
        hp = regressor(mode, model_id)
        
    elif args.model_type== 'classifier':
        if args.transfer:
            hp = classifier_transfered(mode, model_id)
        else:
            hp = classifier(mode, model_id)   

    else:
        raise KeyError
    
        
    hp.features = sample.features
    hp.targets = sample.targets
    hp.device = device

    train_tensor = DynamicHeteroGraphTemporalSignal(train_list)
    test_tensor = DynamicHeteroGraphTemporalSignal(test_list)
    
    if len(test_list)>0:
        heteroData = test_tensor[0]
    else:
        heteroData = train_tensor[0]
    hp.metadata = heteroData.metadata()
    
    print('==========  data information  =========')
    print('data dir', args.data_dir)
    print('test dir', args.test_dir)
    print('number of train, validation, test runs', num_train, num_valid, num_test)
    print('GrainNN frames: ', hp.frames)
    print('features: ', [(k, v) for k, v in hp.features.items()])
    print('targets: ', [(k, v) for k, v in hp.targets.items()])
    print('heteroData metadata', heteroData.metadata())
    print('nodes in samples', [(k, v.shape[0]) for k, v in sample.feature_dicts.items()])
    print('edges in samples', [(k, v.shape[1]) for k, v in sample.edge_index_dicts.items()])
    print('\n')
    
    
    
    print('************ setup model ***********')
    print('==========  architecture  ========')
    print('type -- multilayer heterogeous GCLSTM')
    
    print('input window', hp.window,'; output window', hp.out_win)
    
    print('input feature dimension: ', [(k, len(v)) for k, v in hp.features.items()])
    print('hidden dim (layer size): ', hp.layer_size, \
          '; number of layers (for both encoder and decoder): ', hp.layers)
    print('\n')
    
    
    if mode == 'train':
        print('************ training specification ***********')            
        print('epochs: ', hp.epoch, '; learning rate: ', hp.lr)
        print('batch size: ', hp.batch_size)
        print('model type: ', args.model_type)
        if args.model_type== 'classifier':
            print('weight of positive event: ', hp.weight)
        print('schedueler step: ', hp.decay_step)
        
        print('\n')
    

   # print(model)
   # for model_id, (name, param) in enumerate(model.named_parameters()):
   #            print(name, model_id)


   # model = DataParallel(model)
        
    
   # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
   # print('total number of trained parameters ', pytorch_total_params)
   # print('\n')
    

    if args.mode == 'train': 
        ## train the model
        train_loader = DataLoader(train_tensor, batch_size=hp.batch_size, shuffle=True)
        test_loader = DataLoader(test_tensor, batch_size=64, shuffle=False)
        train_loss_list=[]
        test_loss_list=[]
        test_auc_list = []
        test_acc_list = []
        start = time.time()
        
        
        if args.model_type== 'regressor':
            model = GrainNN_regressor(hp)
        if args.model_type== 'classifier':
            
            if args.transfer:
                regressor_id = 17
                hp_r = regressor(mode, regressor_id)
                hp_r.features = sample.features
                hp_r.targets = sample.targets
                hp_r.device = device
                hp_r.metadata = heteroData.metadata()
                pretrained_model = GrainNN_regressor(hp_r)
                pretrained_model.load_state_dict(torch.load('./GR/regressor'+str(regressor_id)))
                pretrained_model.eval()
                print('transfered learned parameters from regressor')
                model = regressor_classifier(hp, pretrained_model)
            else:
                model = GrainNN_classifier(hp)
        
        model = train(model, train_loader, test_loader)
        x = train.x
        y = train.y
        edge = train.edge
        end = time.time()
        print('training time', end - start)
        
        
        fig, ax = plt.subplots() 
        ax.semilogy(train_loss_list)
        ax.semilogy(test_loss_list)
        if args.model_type== 'classifier':
            ax2 = ax.twinx()
            ax2.plot(test_auc_list, c='r')
            ax2.set_ylabel('PRAUC')
            ax2.legend(['PRAUC'],loc='upper center')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend(['training loss', 'validation loss'])
        plt.title('training time:'+str( "%d"%int( (end-start)/60 ) )+'min')
        plt.savefig('loss.png',dpi=600, bbox_inches='tight')
        
        if args.model_type== 'classifier':
        
            fig, ax = plt.subplots() 
            ax.scatter(train.rlist, train.plist)
            ax.set_ylim(bottom=0.)
            ax.set_xlim(left=0.)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Plot')
            plt.savefig('PR.png',dpi=600, bbox_inches='tight')        
        
            optim_arg = max(range(len(train.plist)), key=lambda i: train.rlist[i]+train.plist[i])
            optim_threshold, optim_p, optim_r = 1 - optim_arg/(len(train.plist)-1), train.plist[optim_arg], train.rlist[optim_arg]
            print('the optimal threshold for classification is: ', optim_threshold, ', with precision/recall', float(optim_p), float(optim_r))

            model.threshold = optim_threshold

        with open('loss.txt', 'w') as f:
            f.write('epoch, training loss, validation loss\n' )
            for i in range(len(train_loss_list)):
                f.write("%d  %f  %f\n"%(i, train_loss_list[i], test_loss_list[i]))
                
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)                
        torch.save(model.state_dict(), args.model_dir + args.model_type + str(model_id))



    if args.mode == 'test':
        
        """
        load model
        """
        assert args.model_type == 'regressor', "default regression"
        
        model.load_state_dict(torch.load(args.model_dir + 'regressor' + str(args.models[0])))
        model.eval() 
        
        hp_classifier = classifier(mode, args.models[1])
        hp_classifier.features = hp.features
        hp_classifier.device = hp.device
        Classifier = GrainNN_classifier(hp_classifier)
        
        Classifier.load_state_dict(torch.load(args.model_dir + 'classifier' + str(args.models[1])))
        Classifier.eval() 

        
        for case, data in enumerate(test_tensor):
            print('case %d'%case)
         #   print(pred['joint'])
          #  traj = graph_trajectory(seed = data.physical_params['seed'], frames = 5)
           # traj.load_trajectory(rawdat_dir = '.')
            with open('./edge_data/traj1.pkl', 'rb') as inp:  
                try:
                    traj = dill.load(inp)
                except:
                    raise EOFError


            for frame in range(hp.frames):
                """
                <1> combine two predictions
                """            
    
                pred = model(data.x_dict, data.edge_index_dict)
                pred_c = Classifier(data.x_dict, data.edge_index_dict)
                pred.update(pred_c)
                
                """
                <2>  update node features
                """
                
                data.x_dict = model.update(data.x_dict, pred)
                
                """
                <3> predict events and update features and connectivity
                """            
                
                data.x_dict, data.edge_index_dict = Classifier.update(data.x_dict, data.edge_index_dict, pred)

     
                """
                Evaluation
                """
                pp_err, pq_err = edge_error_metric(data.edge_index_dict, data['nxt'])
                
                
                traj.GNN_update( (data.x_dict['joint'][:,:2]).detach().numpy())
               # traj.show_data_struct()
                
                
                print('connectivity error of the graph: pp edge %f, pq edge %f'%(pp_err, pq_err))
              #  print('case %d the error %f at sampled height %d'%(case, traj.error_layer, 0))
            

    
