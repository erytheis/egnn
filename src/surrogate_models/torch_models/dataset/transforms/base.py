import copy
from typing import List, Union

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, HeteroData

from src.utils.torch.torch_utils import get_symmetrically_normalized_adjacency, IterableCompose
from src.utils.utils import Iterator
from src.utils.wds_utils import get_hydraulic_resistance


class BaseTransform(torch_geometric.transforms.BaseTransform):

    def __init__(self):
        self.run_counter = 0
        self.indices_inferred = False
        self.invertible = True


    def __call__(self, *args, **kwargs):
        ret = self.forward(*args, **kwargs)
        self.run_counter += 1
        return ret

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def infer_parameters(self, *args, **kwargs):
        self._infer_parameters(*args, **kwargs)
        self.indices_inferred = True
        pass

    def _infer_parameters(self, *args, **kwargs):
        pass

    def inverse(self, data, *args, **kwargs):
        return data

    def copy(self):
        return copy.deepcopy(self)


def plot_x_histogram(x, title, bins=100):
    import matplotlib.pyplot as plt
    plt.hist(x.numpy().flatten(), bins=bins)
    plt.title(title)
    plt.show()


class Compose(torch_geometric.transforms.Compose, IterableCompose):
    transforms: List[BaseTransform]

    def __init__(self, transforms):
        super().__init__(transforms)

    def infer_parameters(self, data):
        for transform in self.transforms:
            if not isinstance(transform, BaseTransform):
                continue
            if transform.indices_inferred:
                continue
            if isinstance(data, (list, tuple)):
                [transform.infer_parameters(d) for d in data]
            else:
                transform.infer_parameters(data)
        return self

    def get(self, class_name):
        for transform in self.transforms:
            if type(transform).__name__ == class_name:
                return transform

    def insert(self, idx, transform: BaseTransform):
        self.transforms.insert(idx, transform)

    def extend(self, idx, transform):
        assert idx in [0, -1], 'idx should be 0 or -1'
        if idx == 0:
            self.transforms = transform.transforms + self.transforms
        else:
            self.transforms = self.transforms + transform.transforms

    def __call__(
        self,
        data: Union[Data, HeteroData],
        inverse: bool = False,
    ) -> Union[Data, HeteroData]:
        for transform in self.transforms:
            if isinstance(data, (list, tuple)):
                data = [transform(d, inverse= inverse) for d in data]
            else:
                data = transform(data, inverse=inverse)
        return data

    def inverse(self, data):
        for transform in self.transforms:
            if isinstance(transform, BaseTransform):
                data = transform.inverse(data)

    def __iter__(self):
        return Iterator(self)

    @property
    def iterable(self):
        return self.transforms

    def copy(self):
        return copy.deepcopy(self)

    def pop(self, idx):

        if isinstance(idx, int):
            return self.transforms.pop(idx)
        elif isinstance(idx, list):
            return [self.transforms.pop(i) for i in idx]


"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
"""


def get_propagation_matrix(x, edge_index, edge_weight, n_nodes):
    # Initialize all edge weights to ones if the graph is unweighted)
    edge_weight = edge_weight if edge_weight else torch.ones(edge_index.shape[1]).to(edge_index.device)
    edge_index, edge_weight = get_symmetrically_normalized_adjacency(edge_index, n_nodes=n_nodes)
    adj = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)

    return adj


class RandomPermuteNodes(BaseTransform):

    def __init__(self):
        super().__init__()

    def forward(self, data):
        """
        Randomly permutes nodes within each disconnected graph separately.
        Indices of the nodes are defined  in data.slices
        """
        if not hasattr(data, 'x'):
            raise ValueError('Data should have x attribute')
        if not hasattr(data, 'edge_index'):
            raise ValueError('Data should have edge_index attribute')
        if hasattr(data, 'ptr'):
            ranges = [
                (i, j) for i, j in zip(data.ptr.tolist(), data.ptr[1:].tolist())
            ]
        else:
            ranges = [(0, data.x.shape[0])]
        permu = torch.cat([torch.arange(i, j)[torch.randperm(j - i)] for i, j in ranges])

        n_nodes = data.x.size(0)
        inits = torch.arange(n_nodes)
        # For the edge_index to work, this must be an inverse permutation map.
        translation = {k: v for k, v in zip(permu.tolist(), inits.tolist())}

        permuted = data.clone()
        permuted.x = permuted.x[permu]
        # Below is the identity transform, by construction of our permutation.
        if data.batch is not None:
            permuted.batch = permuted.batch[permu]
        permuted.edge_index = (
            permuted.edge_index.cpu()
                .apply_(translation.get)
                .to(data.edge_index.device)
        )

        # Permute edge_attr as well
        if hasattr(data, 'edge_attr'):
            permuted.edge_attr = permuted.edge_attr[permu]

        return data


class HazenWilliamsWeights(BaseTransform):
    def __init__(self, remove_original=False, log=False, log_base=None, invert=False):
        super().__init__()
        self.log = log
        self.log_base = log_base
        self.remove_original = remove_original
        self.loss_coefficient_index = None
        self.invert = False # invert resistance

    def forward(self, data, inverse=False, *args, **kwargs):
        if inverse:
            return self.inverse(data)
        return data

    def _infer_parameters(self, data, inverse=False):



        key = 'edge_attr' if hasattr(data, 'edge_attr') else 'x'

        edge_attributes = getattr(data, key)
        edge_attr_names = getattr(data, f'{key}_names')

        if 'diameter' not in edge_attr_names:
            raise ValueError('Edge attributes should have diameter attribute')
        if 'length' not in edge_attr_names:
            raise ValueError('Edge attributes should have length attribute')
        if 'roughness' not in edge_attr_names:
            raise ValueError('Edge attributes should have roughness attribute')

        # get_hazen_williams  headloss  #TODO place it somewhere else
        length = edge_attributes[:, edge_attr_names.index('length')]
        roughness = edge_attributes[:, edge_attr_names.index('roughness')]
        diameter = edge_attributes[:, edge_attr_names.index('diameter')]
        loss_coefficient = get_hydraulic_resistance(length, diameter, roughness)

        if self.remove_original:
            edge_attributes = edge_attributes[:, ~np.isin(edge_attr_names, ['length', 'diameter', 'roughness'])]
            edge_attr_names = [name for name in edge_attr_names if
                               name not in ['length', 'diameter', 'roughness']]

        edge_attr_names.append('loss_coefficient')

        print(data.wds_names[0], loss_coefficient)

        if self.invert:
            loss_coefficient = 1 / loss_coefficient

        if self.log:
            if self.log_base is None:
                loss_coefficient = torch.log(loss_coefficient)
            elif self.log_base == 10:
                loss_coefficient = torch.log10(loss_coefficient)

        loss_coefficient[loss_coefficient == -np.inf] = loss_coefficient[loss_coefficient != -np.inf].min()
        loss_coefficient[loss_coefficient == np.inf] = loss_coefficient[loss_coefficient != np.inf].max()
        #
        edge_attributes = torch.cat([edge_attributes, loss_coefficient.unsqueeze(1)], dim=1)

        self.loss_coefficient_index = len(edge_attr_names) - 1

        setattr(data, key, edge_attributes)
        setattr(data, f'{key}_names', edge_attr_names)

    #
    def inverse(self, batch, inplace=False):
        edge_attr = batch.edge_attr
        if not inplace:
            edge_attr = batch.edge_attr.clone()

        if self.log:
            if self.log_base is None:
                edge_attr[:,self.loss_coefficient_index] = torch.exp(edge_attr[:,self.loss_coefficient_index])
            elif self.log_base == 10:
                edge_attr[:,self.loss_coefficient_index] = torch.pow(10, edge_attr[:,self.loss_coefficient_index])

        if self.invert:
            edge_attr[:,self.loss_coefficient_index] = 1 / edge_attr[:,self.loss_coefficient_index]

        batch.edge_attr = edge_attr
        return batch





class NodeDegree(BaseTransform):
    """
    Adds a one-hot encoding of the node degree to node features
    """

    def __init__(self, max_degree=4):
        super().__init__()
        self.max_degree = max_degree

    def forward(self, batch):
        # one hot encode the degree
        key = 'x'

        x = getattr(batch, key)
        edge_index = batch.edge_index
        n_nodes = batch.x.shape[0]

        degrees = torch.zeros(n_nodes, self.max_degree, device=x.device)
        degree = torch.bincount(edge_index[0], minlength=n_nodes) + torch.bincount(edge_index[1], minlength=n_nodes)
        degrees[torch.arange(n_nodes), degree - 1] = 1

        # assign the one hot encoding to the node features to the rightmost columns
        x[:, -self.max_degree:] = degrees
        batch.x = x
        return batch

    def _infer_parameters(self, data, *args, **kwargs):
        """
        Extends the node features by the number of maximum node degree
        :return:
        """
        key = 'x'
        new_feature = torch.zeros((data[key].shape[0], self.max_degree), dtype=torch.float, device=data[key].device)
        data[key] = torch.cat([data[key], new_feature], dim=1)
        data['{}_names'.format(key)].extend(['degree_{}'.format(i) for i in range(self.max_degree)])
