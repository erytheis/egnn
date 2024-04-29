import json
from typing import Optional, List, Iterable, Union, Tuple
# from prettytable import PrettyTable
from itertools import accumulate
from bisect import bisect
from random import random

import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import torch_geometric
from prettytable import PrettyTable
from torch import Tensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from torch_scatter import scatter, segment_csr, gather_csr

from src.utils.utils import Iterator
from src.surrogate_models.torch_models.visualization.writer.writers import BaseWriter


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use, device_name='cuda:0'):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device(device_name if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer: Optional[BaseWriter] = None, filterby: Optional[List[str]] = None, validation=False):
        self.writer = writer
        self.validation = validation
        if validation:
            keys = [f'val/{k}' for k in keys]
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1, log=True):
        if self.validation:
            key = f'val/{key}'
        if self.writer is not None and log:
            self.writer.log({key: value})
        # if hasattr(value, "__len__"):
        #     return
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def log(self):
        if self.writer is not None:
            self.writer.log(self._data.average.to_dict())

    def update_metrics(self, metrics, log=True):
        for key, value in metrics.items():
            self.update(key, value, log=False)
        if log and self.writer is not None:
            metrics = {f'val/{k}': v for k, v in metrics.items()}
            self.writer.log(metrics)

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def count_parameters(model, print_architecture=False):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    if print_architecture:
        print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def fully_adjacent():
    root_indices = torch.nonzero(roots, as_tuple=False).squeeze(-1)
    target_roots = root_indices.index_select(dim=0, index=batch)
    source_nodes = torch.arange(0, data.num_nodes).to(self.device)
    edges = torch.stack([source_nodes, target_roots], dim=0)


def get_symmetrically_normalized_adjacency(edge_index, n_nodes):
    """
    Given an edge_index, return the same edge_index and edge weights computed as
    \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}.
    """
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=n_nodes)
    deg += scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, DAD


"""
Diffusion models utils
"""


def get_rw_adj(edge_index, edge_weight=None, norm_dim=1, fill_value=0., num_nodes=None, dtype=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)

    if not fill_value == 0:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    indices = row if norm_dim == 0 else col
    deg = scatter_add(edge_weight, indices, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-1)
    edge_weight = deg_inv_sqrt[indices] * edge_weight if norm_dim == 0 else edge_weight * deg_inv_sqrt[indices]
    return edge_index, edge_weight


def gcn_norm_fill_val(edge_index, edge_weight=None, fill_value=0., num_nodes=None, dtype=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)

    if not int(fill_value) == 0:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


# Counter of forward and backward passes.
class Meter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.sum = 0
        self.cnt = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.cnt += 1

    def get_average(self):
        if self.cnt == 0:
            return 0
        return self.sum / self.cnt

    def get_value(self):
        return self.val


# https://twitter.com/jon_barron/status/1387167648669048833?s=12
# @torch.jit.script
def squareplus(src: Tensor, index: Optional[Tensor], ptr: Optional[Tensor] = None,
               num_nodes: Optional[int] = None) -> Tensor:
    r"""Computes a sparsely evaluated softmax.
      Given a value tensor :attr:`src`, this function first groups the values
      along the first dimension based on the indices specified in :attr:`index`,
      and then proceeds to compute the softmax individually for each group.
      Args:
          src (Tensor): The source tensor.
          index (LongTensor): The indices of elements for applying the softmax.
          ptr (LongTensor, optional): If given, computes the softmax based on
              sorted inputs in CSR representation. (default: :obj:`None`)
          num_nodes (int, optional): The number of nodes, *i.e.*
              :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
      :rtype: :class:`Tensor`
      """
    out = src - src.max()
    # out = out.exp()
    out = (out + torch.sqrt(out ** 2 + 4)) / 2

    if ptr is not None:
        out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
    else:
        raise NotImplementedError

    return out / (out_sum + 1e-16)


class MaxNFEException(Exception): pass




class IterableCompose:
    iterable: Iterable

    def __iter__(self):
        return Iterator(self)

    def __getitem__(self, index):
        return self.iterable[index]


def random_choices(weights=None, *, cum_weights=None, k =1, n=None):
    """Return a k sized list of population elements chosen with replacement.

    If the relative weights or cumulative weights are not specified,
    the selections are made with equal probability.

    """

    if cum_weights is None:
        cum_weights = list(accumulate(weights))
    elif weights is not None:
        raise TypeError('Cannot specify both weights and cumulative weights')
    if len(cum_weights) != n:
        raise ValueError('The number of weights does not match the population')
    total = cum_weights[-1] + 0.0   # convert to float
    hi = n - 1
    indices = [bisect(cum_weights, random() * total, 0, hi)
            for i in repeat(None, k)]

    return indices


def to_undirected(
    edge_index: Tensor,
    edge_attr: Union[Optional[Tensor], List[Tensor]] = None,
    num_nodes: Optional[int] = None,
    reduce: str = "add",
    sort_by_row: bool = False,
    coalesce: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, List[Tensor]]]:
    r"""Converts the graph given by :attr:`edge_index` to an undirected graph
    such that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
            dimensional edge features.
            If given as a list, will remove duplicates for all its entries.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        reduce (string, optional): The reduce operation to use for merging edge
            features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`). (default: :obj:`"add"`)

    :rtype: :class:`LongTensor` if :attr:`edge_attr` is :obj:`None`, else
        (:class:`LongTensor`, :obj:`Tensor` or :obj:`List[Tensor]]`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1],
        ...                            [1, 0, 2]])
        >>> to_undirected(edge_index)
        tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]])

        >>> edge_index = torch.tensor([[0, 1, 1],
        ...                            [1, 0, 2]])
        >>> edge_weight = torch.tensor([1., 1., 1.])
        >>> to_undirected(edge_index, edge_weight)
        (tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]]),
        tensor([2., 2., 1., 1.]))

        >>> # Use 'mean' operation to merge edge features
        >>>  to_undirected(edge_index, edge_weight, reduce='mean')
        (tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]]),
        tensor([1., 1., 1., 1.]))
    """
    # Maintain backward compatibility to `to_undirected(edge_index, num_nodes)`
    if isinstance(edge_attr, int):
        edge_attr = None
        num_nodes = edge_attr

    row, col = edge_index[0], edge_index[1]
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    if isinstance(edge_attr, Tensor):
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    elif isinstance(edge_attr, (list, tuple)):
        edge_attr = [torch.cat([e, e], dim=0) for e in edge_attr]
    if coalesce:
        return torch_geometric.utils.coalesce(edge_index, edge_attr, num_nodes, reduce, sort_by_row=sort_by_row)
    else:
        return edge_index, edge_attr