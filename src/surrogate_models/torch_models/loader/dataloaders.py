from typing import Union, List

import torch
from torch.utils.data._utils.collate import default_collate

from torch_geometric.data import Dataset, Data, HeteroData, Batch
from torchvision import datasets, transforms
from collections import abc as container_abcs

from src.surrogate_models.torch_models.base.base_dataloader import Collater
from src.surrogate_models.torch_models.loader.base_dataloader import BaseGNNDataLoader, BaseDataLoader, OLDDataLoader

# from line_profiler_pycharm import profile

int_classes = int


class MnistDataLoader(BaseGNNDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class WDSDataLoader(BaseGNNDataLoader):

    def __init__(self,
                 dataset: Union[Dataset, List[Data], List[HeteroData]],
                 batch_size: int = 1,
                 shuffle: bool = False,
                 validation_split: float = 0.1,
                 num_workers: int = 0):
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)


class FillGPULoader:

    def __init__(self, dataset, memory_limit, step, device='cpu'):
        self.dataset = dataset
        self.device = device
        self.memory_limit = memory_limit
        self.step = step
        self.collater = Collater(None, None)

    def __iter__(self):
        # fill gpu
        batch = None
        minibatch_filled = False
        memory_taken = 0
        batch_ = []
        i = 0
        minibatch = []

        while memory_taken < self.memory_limit:
            i = 0
            # first fill the batch
            while len(minibatch) < self.step:

                if i > (len(self.dataset) - 1):
                    i = 0
                    minibatch.extend(minibatch)
                else:
                    minibatch.append(self.dataset[i])
                    i = i + 1

            batch_.extend(minibatch)
            batch = self.collater(batch_)
            memory_taken = (torch.cuda.mem_get_info(self.device)[1] - torch.cuda.mem_get_info(self.device)[
                0]) / 1024 / 1024 / 1024
            print("torch.cuda.memory_allocated: %fGB" % memory_taken)
        print('filled')
        # now yield the batch
        yield batch


class FillBatchLoader:

    def __init__(self, dataset, size_limit, step, key='num_graphs',
                 device='cpu'):
        self.dataset = dataset
        self.device = device
        self.size_limit = size_limit
        self.key = key
        self.step = step
        self.collater = Collater(None, None)
        self.memory_limit = torch.cuda.get_device_properties(self.device).total_memory / 1024 / 1024 / 1024

    def __iter__(self):
        # fill gpu
        length, new_length = 0, 0
        memory_taken = 0
        batch_, minibatch = [], []
        minibatch_memory_taken = None

        while length < self.size_limit and (memory_taken + 1) < self.memory_limit:
            i = 0
            # first fill the minibatch
            while new_length < self.step:

                if i > (len(self.dataset) - 1):
                    i = 0

                    minibatch.extend(minibatch)
                    new_length += getattr(self.dataset[i], self.key)
                else:
                    minibatch.append(self.dataset[i])
                    new_length += getattr(self.dataset[i], self.key)

                    i = i + 1

                # estimate memory taken by a minibatch
                if minibatch_memory_taken is None:
                    temp_batch = self.collater(minibatch)
                    minibatch_memory_taken = (torch.cuda.mem_get_info(self.device)[1] - torch.cuda.mem_get_info(self.device)[
                        0]) / 1024 / 1024 / 1024
                    del temp_batch

            length += new_length
            print('new_length', new_length)
            print('length', length)

            batch_.extend(minibatch)

            memory_taken += minibatch_memory_taken *2

        batch = self.collater(batch_)
        print("ESTIMATED torch.cuda.memory_allocated: %fGB" % memory_taken)
        memory_taken = (torch.cuda.mem_get_info(self.device)[1] - torch.cuda.mem_get_info(self.device)[
                0]) / 1024 / 1024 / 1024
        print("ACTUAL torch.cuda.memory_allocated: %fGB" % memory_taken)
        print('filled')
        # now yield the batch
        yield batch
