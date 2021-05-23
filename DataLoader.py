import torch
from torch.utils.data import SubsetRandomSampler

from HerbariumDataset import HerbariumDataset

import numpy as np


class TrainDataLoader:
    def __init__(self, data: HerbariumDataset, valid_frac=0.2,
                 batch_size=64):
        self._data = data
        self._valid_fac = valid_frac
        self._batch_size = batch_size
    
    def _get_samplers(self):
        num_train = len(self._data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(self._valid_fac * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        return train_sampler, valid_sampler
    
    @property
    def data_loaders(self):
        train_sampler, valid_sampler = self._get_samplers()
        
        train_loader = torch.utils.data.DataLoader(self._data,
                                                   batch_size=self._batch_size,
                                                   sampler=train_sampler,
                                                   num_workers=0)
        valid_loader = torch.utils.data.DataLoader(self._data,
                                                   batch_size=self._batch_size,
                                                   sampler=valid_sampler,
                                                   num_workers=0)
        return train_loader, valid_loader
    

class TestDataLoader:
    def __init__(self, data: HerbariumDataset, batch_size=64):
        self._data = data
        self._batch_size = batch_size
    
    @property
    def data_loader(self):
        test_loader = torch.utils.data.DataLoader(self._data,
                                                  batch_size=self._batch_size,
                                                  num_workers=0)
        return test_loader
