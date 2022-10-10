from torch_geometric.data import HeteroData
from graph_datastruct import graph_trajectory
import torch

if __name__=='__main__':


    data = HeteroData()

    
    data['grain'].x = ... # [num_grains, num_features_grain]
    data['grain'].y = ... 
    
    
    data['joint'].x = ... # [num_joints, num_features_joint]
    data['joint'].y = ... # [num_joints, 2]
    
    
    data['grain', 'push', 'joint'].edge_index = ... # [2, num_edges_grains]
    data['joint', 'pull', 'grain'].edge_index = ... # [2, num_edges_grains]
    data['joint', 'connect', 'joint'].edge_index = ... # [2, num_edges_joints]
    
    
    
  #  data['joint', 'connect', 'joint'].edge_attr = ... # [num_edges_joints, num_features_joints]
    


   # data augmentation











