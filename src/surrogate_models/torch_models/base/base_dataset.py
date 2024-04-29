from typing import List

import torch
from torch.utils.data import WeightedRandomSampler


class ConcatDataset(torch.utils.data.ConcatDataset):
    dataset_types: List[str]


    @property
    def transform(self):
        return self.datasets[0].transform

    @property
    def pre_transform(self):
        return self.datasets[0].pre_transform

    def clear_cache(self):
        for ds in self.datasets:
            ds.clear_cache() if hasattr(ds, 'clear_cache') else None