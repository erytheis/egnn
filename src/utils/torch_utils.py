import json
from typing import Optional, List, Iterable
import numpy as np
import scipy.sparse as sp

import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


from torch import Tensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from torch_scatter import scatter, segment_csr, gather_csr

from src.utils.utils import Iterator
# from src.surrogate_models.torch_models.visualization.writer.writers import BaseWriter


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

    device = torch.device(device_name if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys: object,
                 writer= None, filterby: Optional[List[str]] = None,
                 validation: object = False) -> object:
        self.writer = writer
        self.validation = validation
        if validation:
            keys = [f'val/{k}' for k in keys]
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.validation:
            key = f'val/{key}'
        if self.writer is not None:
            self.writer.log(key, value)
        if hasattr(value, "__len__"):
            return
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)




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


def sparse_eye(size, device=None, value=1, bool_vector=None):
    """
    Returns the identity matrix as a sparse matrix
    """
    indices = torch.arange(0, size, device=device).long().unsqueeze(0).expand(2, size)
    values = torch.tensor(value, device=device).expand(size)

    if bool_vector is not None:
        values = values * bool_vector

    cls = getattr(torch.sparse, values.type().split(".")[-1])
    return cls(indices, values, torch.Size([size, size]))


# implicit layers utils

def get_spectral_rad(sparse_tensor, tol=1e-5):
    """Compute spectral radius from a tensor"""
    A = sparse_tensor.data.coalesce().cpu()
    A_scipy = sp.coo_matrix((np.abs(A.values().numpy()), A.indices().numpy()), shape=A.shape)
    return np.abs(sp.linalg.eigs(A_scipy, k=1, return_eigenvectors=False)[0]) + tol


def slice_torch_sparse_coo_tensor(t, args):
    """
    params:
    -------
    t: tensor to slice
    slices: slice for each dimension

    returns:
    --------
    t[slices[0], slices[1], ..., slices[n]]
    """

    t = t.coalesce()
    assert len(args) == len(t.size())
    for i in range(len(args)):
        if type(args[i]) is not torch.Tensor:
            args[i] = torch.tensor(args[i], dtype=torch.long, device=t.device)

    indices = t.indices()
    values = t.values()
    for dim, slice in enumerate(args):
        invert = False
        if t.size(0) * 0.6 < len(slice):
            invert = True
            all_nodes = torch.arange(t.size(0), device=t.device)
            unique, counts = torch.cat([all_nodes, slice], device=t.device).unique(return_counts=True)
            slice = unique[counts == 1]
        if slice.size(0) > 400:
            mask = ainb_wrapper(indices[dim], slice)
        else:
            mask = ainb(indices[dim], slice)
        if invert:
            mask = ~mask
        indices = indices[:, mask]
        values = values[mask]

    return torch.sparse_coo_tensor(indices, values, device=t.device).coalesce()



def slice_torch_sparse_coo_tensor(indices, values, sizes, args):
    """
    params:
    -------
    t: tensor to slice
    slices: slice for each dimension

    returns:
    --------
    t[slices[0], slices[1], ..., slices[n]]
    """

    for dim, slice in enumerate(args):
        invert = False
        if sizes[0] * 0.6 < len(slice):
            invert = True
            all_nodes = torch.arange(sizes[0], device=indices.device)
            unique, counts = torch.cat([all_nodes, slice]).unique(return_counts=True)
            slice = unique[counts == 1]
        if slice.size(0) > 400:
            mask = ainb_wrapper(indices[dim], slice)
        else:
            mask = ainb(indices[dim], slice)
        if invert:
            mask = ~mask
        indices = indices[:, mask]
        values = values[mask]

    return indices, values


def ainb(a, b):
    """gets mask for elements of a in b"""

    size = (b.size(0), a.size(0))

    if size[0] == 0:  # Prevents error in torch.Tensor.max(dim=0)
        return torch.tensor([False] * a.size(0), dtype=torch.bool)

    a = a.expand((size[0], size[1]))
    b = b.expand((size[1], size[0])).T

    mask = a.eq(b).max(dim=0).values

    return mask


def ainb_wrapper(a, b, splits=.72):
    inds = int(len(a) ** splits)

    tmp = [ainb(a[i * inds:(i + 1) * inds], b) for i in list(range(inds))]

    return torch.cat(tmp)

# #
# indices = L.coalesce().indices()
# values = L.coalesce().values()
#
#
# L__ = L.to_dense()[edges_to_keep][:, edges_to_keep].to_sparse()
# indices__ = L__.coalesce().indices()
# values__ = L__.coalesce().values()
#
#
# for i in range(2):
#     a = indices[i]
#     b = edges_to_keep.nonzero(as_tuple=True)[0]
#
#     size = (b.size(0), a.size(0))
#
#
#     a = a.expand((size[0], size[1]))
#     b = b.expand((size[1], size[0])).T
#
#     mask = a.eq(b).max(dim=0).values
#     indices = indices[:,mask]
#     values = values[mask]
# L_ = torch.sparse_coo_tensor(indices, values).coalesce()


def degree(index: Tensor, num_nodes: Optional[int] = None,
           dtype: Optional[torch.dtype] = None) -> Tensor:
    r"""Computes the (unweighted) degree of a given one-dimensional index
    tensor.

    Args:
        index (LongTensor): Index tensor.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dtype (:obj:`torch.dtype`, optional): The desired data type of the
            returned tensor.

    :rtype: :class:`Tensor`

    Example:

        >>> row = torch.tensor([0, 1, 0, 2, 0])
        >>> degree(row, dtype=torch.long)
        tensor([3, 1, 1])
    """
    N = maybe_num_nodes(index, num_nodes)
    out = torch.zeros((N, ), dtype=dtype, device=index.device)
    one = torch.ones((index.size(0), ), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index, one)