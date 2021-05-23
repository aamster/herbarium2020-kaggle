import torch
from torch.utils.data import SubsetRandomSampler

from HerbariumDataset import HerbariumDataset

import numpy as np


class TrainDataLoader:
    def __init__(self, data: HerbariumDataset, batch_size=64):
        self._data = data
        self._batch_size = batch_size

    @property
    def data_loaders(self):
        train_loader = torch.utils.data.DataLoader(self._data,
                                                   batch_size=self._batch_size,
                                                   shuffle=True,
                                                   num_workers=0)
        valid_loader = torch.utils.data.DataLoader(self._data,
                                                   batch_size=self._batch_size,
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
