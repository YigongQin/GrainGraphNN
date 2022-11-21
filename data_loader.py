#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 21:25:20 2022

@author: yigongqin
"""

import torch
from torch_geometric.data import HeteroData


class DynamicHeteroGraphTemporalSignal(object):
    r"""A data iterator object to contain a dynamic heterogeneous graph with a
    changing edge set and weights. The feature set and node labels
    (target) are also dynamic. The iterator returns a single discrete temporal
    snapshot for a time period (e.g. day or week). This single snapshot is a
    Pytorch Geometric HeteroData object. Between two temporal snapshots the edges,
    edge weights, target matrices and optionally passed attributes might change.
    Args:
        edge_index_dicts (List of dictionaries where keys=Tuples and values=Numpy arrays):
         List of relation type tuples and their edge index tensors.
        edge_weight_dicts (List of dictionaries where keys=Tuples and values=Numpy arrays):
         List of relation type tuples and their edge weight tensors.
        feature_dicts (List of dictionaries where keys=Strings and values=Numpy arrays): List of node
         types and their feature tensors.
        target_dicts (List of dictionaries where keys=Strings and values=Numpy arrays): List of node
         types and their label (target) tensors.
        **kwargs (optional List of dictionaries where keys=Strings and values=Numpy arrays): List
         of node types and their additional attributes.
    """

    def __init__(self, data_list):

        self.data_list = data_list

      #  self._check_temporal_consistency()
      #  self._set_snapshot_count()

    def _check_temporal_consistency(self):
        assert len(self.feature_dicts) == len(
            self.target_dicts
        ), "Temporal dimension inconsistency."
        assert len(self.edge_index_dicts) == len(
            self.edge_weight_dicts
        ), "Temporal dimension inconsistency."
        assert len(self.feature_dicts) == len(
            self.edge_weight_dicts
        ), "Temporal dimension inconsistency."
        for key in self.additional_feature_keys:
            assert len(self.target_dicts) == len(
                getattr(self, key)
            ), "Temporal dimension inconsistency."
            
    def __len__(self):
         return len(self.data_list)
     
    def _set_snapshot_count(self):
        self.snapshot_count = len(self.feature_dicts)

    def _get_edge_index(self):
        if self.edge_index_dicts is None:
            return self.edge_index_dicts
        else:
            return {key: torch.LongTensor(value) for key, value in self.edge_index_dicts.items()
                    if value is not None}

    def _get_mask(self):
        if self.mask is None:
            return self.mask
        else:
            return {key: torch.LongTensor(value) for key, value in self.mask.items()
                    if value is not None}

    def _get_edge_weight(self):
        if self.edge_weight_dicts is None:
            return self.edge_weight_dicts
        else:
            return {key: torch.FloatTensor(value) for key, value in self.edge_weight_dicts.items()
                    if value is not None}

    def _get_features(self):
        if self.feature_dicts is None:
            return self.feature_dicts
        else:
            return {key: torch.FloatTensor(value) for key, value in self.feature_dicts.items()
                    if value is not None}

    def _get_target(self):
        if self.target_dicts is None:
            return self.target_dicts
        else:
            return {key: torch.FloatTensor(value) if value.dtype.kind == "f" else torch.LongTensor(value)
                    if value.dtype.kind == "i" else value for key, value in self.target_dicts.items()
                    if value is not None}

    def _get_additional_feature(self, feature_key: str):
        feature = self.add_feature[feature_key]
        if feature is None:
            return feature
        else:
            return {key: torch.FloatTensor(value) if value.dtype.kind == "f" else torch.LongTensor(value)
                    if value.dtype.kind == "i" else value for key, value in feature.items()
                    if value is not None}

    def _get_additional_features(self):
        additional_features = {
            key: self._get_additional_feature(key)
            for key in self.add_feature.keys()
        }
        return additional_features

    def __getitem__(self, index):
        
        data = self.data_list[index]
        self.edge_index_dicts = data.edge_index_dicts
        self.edge_weight_dicts = data.edge_weight_dicts
        self.feature_dicts = data.feature_dicts
        self.target_dicts = data.target_dicts
        self.mask = data.mask
        self.add_feature = data.additional_features
 
        edge_index_dict = self._get_edge_index()
        edge_weight_dict = self._get_edge_weight()        
        x_dict = self._get_features()
        y_dict = self._get_target()
        mask = self._get_mask()
        
        additional_features = self._get_additional_features()
   
        snapshot = HeteroData()
        if x_dict:
            for key, value in x_dict.items():
                snapshot[key].x = value
        if edge_index_dict:
            for key, value in edge_index_dict.items():
                snapshot[key].edge_index = value
        if edge_weight_dict:
            for key, value in edge_weight_dict.items():
                snapshot[key].edge_attr = value
        if y_dict:
            for key, value in y_dict.items():
                snapshot[key].y = value
        if mask:
            for key, value in mask.items():
                snapshot['mask'][key] = value            

        if additional_features:
     
            for feature_name, feature_dict in additional_features.items():
                if feature_dict:
             
                    for key, value in feature_dict.items():
                        snapshot[feature_name][key] = value
                
        snapshot.physical_params = data.physical_params
        return snapshot

    def __next__(self):
        if self.t < len(self.data_list):
            snapshot = self[self.t]
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self