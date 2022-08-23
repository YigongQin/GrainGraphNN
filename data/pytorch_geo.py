#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 22:02:10 2022

@author: yigongqin
"""

import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset, ShapeNet
import networkx as nx
from torch_geometric.loader import DataLoader

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
data = dataset[0]
print(data.is_undirected())
loader = DataLoader(dataset, batch_size=32, shuffle=True)

g = torch_geometric.utils.to_networkx(data, to_undirected=True)
nx.draw(g)





'''
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
g1 = torch_geometric.utils.to_networkx(data, to_undirected=True)
nx.draw(g1)
'''