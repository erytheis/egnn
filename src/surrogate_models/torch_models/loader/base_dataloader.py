import json
from typing import List, Union, Mapping, Sequence, Optional

import numpy as np
import torch.utils.data.sampler
import torch_geometric
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler, SequentialSampler
from torch_geometric.data import HeteroData, Data, Dataset, Batch
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.surrogate_models.torch_models.data.simplex import SimplexData
from src.utils.utils import NpEncoder


class Collater(torch_geometric.loader.dataloader.Collater):

    def __call__(self, batch):
        elem = batch[0]
        if isinstance(elem, SimplexData):
            return elem.from_data_list(batch, self.follow_batch,
                                       self.exclude_keys)
        elif isinstance(elem, BaseData):
            return Batch.from_data_list(batch, self.follow_batch,
                                        self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f'DataLoader found invalid type: {type(elem)}')


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    train_datasets: List
    validation_datasets: List

    def __init__(self, dataset,
                 batch_size,
                 shuffle,
                 validation_split,
                 num_workers,
                 test_split=0.0,
                 collate_fn=default_collate,
                 to_ann: bool = False,
                 context: str = 'spawn',
                 loaded_indices: Optional[dict] = None,
                 ):
        self.validation_split = validation_split
        self.test_split = test_split
        self.shuffle = shuffle
        self.to_ann = to_ann

        self.batch_idx = 0
        self.n_samples = len(dataset)

        samplers = self._split_sampler(self.validation_split, self.test_split, dataset,
                                       loaded_indices)
        self.sampler, self.valid_sampler, self.test_sampler = samplers

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            # 'spawn_processes': context == 'spawn'
        }

        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, validation_split, test_split, dataset, loaded_indices=None):

        if loaded_indices is None:

            indices = {'train': [],
                       'validation': [],
                       'test': []
                       }
            lens = {k:0 for k in indices.keys()}

            # Add dedicated subset to each dataset type
            if isinstance(dataset, torch.utils.data.ConcatDataset):
                _start_idx = 0
                for ds, ds_type in zip(dataset.datasets, dataset.types):
                    _end_idx = len(ds) + _start_idx

                    # add indices and limit the size
                    subset_indices = np.arange(_start_idx, _end_idx)

                    # if ds_type == 'validation':
                    if self.sample_fraction is not None:
                        limit = int((_end_idx - _start_idx) * self.sample_fraction )
                        if limit > 10:
                            subset_indices = subset_indices[:limit]

                    indices[ds_type].extend(subset_indices)
                    _start_idx = _end_idx
            else:
                indices['train'].extend(np.arange(self.n_samples))

            idx_full = indices['train']
            valid_idx, test_idx = [], []

            # configure the maximum sizes of each type
            for ds_type in indices.keys():
                if not hasattr(self, f'{ds_type}_split'):
                    continue
                _split = getattr(self, f'{ds_type}_split')
                if isinstance(_split, int):
                    assert _split > 0
                    lens[ds_type] = _split
                else:
                    lens[ds_type] = int(len(idx_full) * _split)

            # get validation indices
            if self.shuffle:
                np.random.shuffle(indices['validation'])
                np.random.shuffle(idx_full)

            if validation_split > 0.0 or len(indices['validation']) > 0:
                valid_idx = idx_full[0:lens['validation']]  # adds fraction from the total set
                valid_idx += indices['validation']

            # get test indices
            if test_split > 0.0 or len(indices['validation']) > 0:

                test_idx = idx_full[lens['validation']:lens['test'] + lens['validation']]  # adds fraction from the total set
                # test_idx += indices['validation'][lens['valid']:]  # adds everything  else from the validation set
                test_idx += indices['test']  # adds everything   from the test set

            train_idx = np.delete(idx_full, np.arange(0, lens['validation']+lens['test']))

        else:
            train_idx = np.array(loaded_indices['train'])
            valid_idx = np.array(loaded_indices.get('validation', []))
            test_idx = np.array(loaded_indices.get('test', []))
            print(f"Loaded indices: {len(train_idx)} train, {len(valid_idx)} validation, {len(test_idx)} test")

        train_sampler = SubsetRandomSampler(train_idx, generator=torch.Generator())
        valid_sampler = SubsetRandomSampler(valid_idx, generator=torch.Generator()) if len(valid_idx) > 0 else None
        test_sampler = SubsetRandomSampler(test_idx, generator=torch.Generator()) if len(test_idx) > 0 else None

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler, test_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return self.__class__.__init__(sampler=self.valid_sampler, **self.init_kwargs)

    @property
    def datasets(self):
        return self.dataset.datasets

    def dump_state(self, filename):
        indices = {'loaded_indices':
                       {'train': list(self.sampler.indices),
                        'validation': list(self.valid_sampler.indices) if self.valid_sampler is not None else [],
                        'test': list(self.test_sampler.indices) if self.test_sampler is not None else []
                        }}
        with open(filename, 'w') as f:
            json.dump(indices, f, cls=NpEncoder)


class BaseGNNDataLoader(BaseDataLoader, PyGDataLoader):
    def __init__(self,
                 dataset: Union[Dataset, List[Data], List[HeteroData]],
                 batch_size: int = 1,
                 shuffle: bool = False,
                 validation_split: float = 0.1,
                 collate_fn=Collater(None, None),
                 num_workers: int = 0,
                 test_split: float = 0.0,
                 loaded_indices: Optional[dict] = None,
                 sample_fraction = None,
                 *args,
                 **kwargs):
        if batch_size == -1:
            batch_size = len(dataset)

        self.sample_fraction =sample_fraction
        self.validation_split = validation_split
        self.test_split = test_split
        self.shuffle = shuffle

        self.n_samples = len(dataset)

        samplers = self._split_sampler(self.validation_split, test_split, dataset, loaded_indices)
        self.sampler, self.valid_sampler, self.test_sampler = samplers

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            'collate_fn': collate_fn,
        }
        PyGDataLoader.__init__(self,
                               sampler=self.sampler,
                               **self.init_kwargs.copy())

        self.collate_fn = collate_fn

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            valid_data_laoder = PyGDataLoader(sampler=self.valid_sampler, **self.init_kwargs)
            valid_data_laoder.collate_fn = self.collate_fn
            return valid_data_laoder

    def split_test(self):
        if self.test_sampler is None:
            return None
        else:
            test_data_laoder = PyGDataLoader(sampler=self.test_sampler, **self.init_kwargs)
            test_data_laoder.collate_fn = self.collate_fn
            return test_data_laoder


class OLDDataLoader(BaseDataLoader, PyGDataLoader):
    def __init__(self,
                 dataset: Union[Dataset, List[Data], List[HeteroData]],
                 batch_size: int = 1,
                 shuffle: bool = False,
                 validation_split: float = 0.1,
                 collate_fn=Collater(None, None),
                 num_workers: int = 0,
                 test_split: float = 0.0,
                 loaded_indices: Optional[dict] = None,
                 *args,
                 **kwargs):
        if batch_size == -1:
            batch_size = len(dataset)

        self.validation_split = validation_split
        self.test_split = test_split
        self.shuffle = shuffle

        self.n_samples = len(dataset)

        samplers = self._split_sampler(self.validation_split, test_split, dataset, loaded_indices)
        self.sampler, self.valid_sampler, self.test_sampler = samplers

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            'collate_fn': collate_fn,
        }
        PyGDataLoader.__init__(self,
                               sampler=self.sampler,
                               **self.init_kwargs.copy())

        self.collate_fn = collate_fn

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            valid_data_laoder = PyGDataLoader(sampler=self.valid_sampler, **self.init_kwargs)
            valid_data_laoder.collate_fn = self.collate_fn
            return valid_data_laoder

    def split_test(self):
        if self.test_sampler is None:
            return None
        else:
            test_data_laoder = PyGDataLoader(sampler=self.test_sampler, **self.init_kwargs)
            test_data_laoder.collate_fn = self.collate_fn
            return test_data_laoder
