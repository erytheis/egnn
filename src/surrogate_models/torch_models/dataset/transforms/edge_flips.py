from typing import List, Optional

import torch
from torch_geometric.data import Data

from src.surrogate_models.torch_models.dataset.transforms.base import BaseTransform


class FlipEdges(BaseTransform):
    """
    Base class for edge flipping transforms, takes as input the indices of edges to flip
    """
    keys: List[str] = ['edge_attr']

    def __init__(self, columns: Optional[List[str]] = None, weight=False):
        self.columns = columns
        self.weight = weight
        self.columns_idx = {}
        super().__init__()

    def forward(self, data: Data, to_flip, inplace=True) -> Data:
        # flip edges where demand > 0
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        if not inplace:
            edge_index = edge_index.clone()

        edge_index[:, to_flip] = torch.flip(edge_index[:, to_flip], [0])
        edge_rows = edge_index[0, to_flip]

        # flip all indices
        for key, value in data.items():
            if 'weight' in key:
                data[key] = data[key][to_flip] * -1

        for i in self.columns_idx['edge_attr']:
            edge_attr[to_flip, i] = edge_attr[to_flip, i] * -1

        data.edge_index = edge_index
        data.edge_attr = edge_attr

        return data

    def _infer_parameters(self, data: Data):
        # names = getattr(data, f'{key}_names')

        for key in self.keys:
            names = data[f'{key}_names']

            self.columns_idx[key] = []

            if data[key] is None:
                continue

            if self.columns is None:
                self.columns = names

            for col in self.columns:
                if col in names:
                    self.columns_idx[key].append(names.index(col))


class AlignEdges(FlipEdges):
    """
    Flips negative edges according to the value in edge_attr in order to make all edge_attr positive
    """

    def forward(self, data: Data, **kwargs) -> Data:
        # flip edges where demand > 0
        to_flip = torch.where(data.edge_attr[:, self.columns_idx['edge_attr']] < 0)[0]
        return super().forward(data, to_flip, )


class FlipAllEdges(FlipEdges):
    """
    Flips negative edges according to the value in edge_attr in order to make all edge_attr positive
    """

    def forward(self, data: Data, **kwargs) -> Data:
        data.edge_index = torch.flip(data.edge_index, [0])
        data.edge_attr[:, self.columns_idx['edge_attr']] = -1
        return data


class RandomFlipEdges(FlipEdges):
    """
    Flips edges randomly and changes the sign of columns of edge_attr
    """

    def __init__(self, columns: Optional[List[str]] = None, p: float = 0.5):
        self.p = p
        super().__init__(columns)

    def forward(self, data: Data, **kwargs) -> Data:
        # Flip random edges with the probability p and the sign of the corresponding edge_attr
        to_flip = torch.where(torch.rand(data.edge_index.shape[1]) < self.p)[0]
        return super().forward(data, to_flip)


class OrderEdges(FlipEdges):
    """
    Flips edges according to the index of the node in edge_index
    """

    def forward(self, data: Data, **kwargs) -> Data:
        # Flip random edges with the probability p and the sign of the corresponding edge_attr
        to_flip = torch.where(data.edge_index[0] > data.edge_index[1])[0]
        return super().forward(data, to_flip)
