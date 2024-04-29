from collections import defaultdict
from typing import Any, List, Union, Optional, Tuple, Mapping

import torch_geometric
from line_profiler_pycharm import profile
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.data.collate import _collate, repeat_interleave, cumsum
from torch_geometric.data.data import BaseData
from torch_geometric.data.dataset import IndexType
from torch_geometric.data.storage import BaseStorage, NodeStorage
from torch_geometric.typing import OptTensor
import torch
import numpy as np


#
#
# class GlobalStorage(torch_geometric.data.storage.GlobalStorage):
#     @property
#     def wds_names(self):
#         return self.wds


class GraphData(torch_geometric.data.Data):
    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None,
                 edge_attr: OptTensor = None, y: OptTensor = None,
                 pos: OptTensor = None, **kwargs):

        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)

        self.x_names = kwargs.get('x_names', None)

        if 'wds_names' in kwargs.keys():
            setattr(self._store, 'wds_names', kwargs['wds_names'])
            self.wds_names = kwargs['wds_names']

    @classmethod
    def from_data_list(cls, data_list: List[BaseData],
                       follow_batch: Optional[List[str]] = None,
                       exclude_keys: Optional[List[str]] = None):
        r"""Constructs a :class:`~torch_geometric.data.Batch` object from a
        Python list of :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` objects.
        The assignment vector :obj:`batch` is created on the fly.
        In addition, creates assignment vectors for each key in
        :obj:`follow_batch`.
        Will exclude any keys given in :obj:`exclude_keys`."""

        batch, slice_dict, inc_dict = collate(
            Batch,
            data_list=data_list,
            increment=True,
            add_batch=not isinstance(data_list[0], Batch),
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        batch._num_graphs = len(data_list)
        batch._slice_dict = slice_dict
        batch._inc_dict = inc_dict

        return batch

    @property
    def num_nodes(self) -> Optional[int]:
        r"""Returns the number of nodes in the graph.

        .. note::
            The number of nodes in the data object is automatically inferred
            in case node-level attributes are present, *e.g.*, :obj:`data.x`.
            In some cases, however, a graph may only be given without any
            node-level attributes.
            PyG then *guesses* the number of nodes according to
            :obj:`edge_index.max().item() + 1`.
            However, in case there exists isolated nodes, this number does not
            have to be correct which can result in unexpected behaviour.
            Thus, we recommend to set the number of nodes in your data object
            explicitly via :obj:`data.num_nodes = ...`.
            You will be given a warning that requests you to do so.
        """
        return sum([v.num_nodes for v in self.node_stores])

    @property
    def wds_names(self) -> Any:
        return self['wds_names'] if 'wds_names' in self._store else None

    @property
    def slices(self):
        return self._store._slice_dict

    def mask_by_key(self, attribute, value):
        """
        Masks internal values in accordance with the
        Args:
            attribute: (str) attribute to return
            value: mask values that are NOT equal to this parameter
        :
        """
        slices = self._store._slice_dict
        mask = np.array(self.wds_names) == value
        start_idx = slices[attribute][:-1][mask]
        end_idx = slices[attribute][1:][mask]
        mask = torch.zeros_like(self[attribute], dtype=torch.bool)
        for l, r in zip(start_idx, end_idx):
            mask[l:r] = 1
        return mask

    def mask_by_features(self, attribute, column_idx=-1):
        """
        Masks internal values in accordance with the
        Args:
            attribute: (str) attribute to return
            column_idx: (int) column index of the feature to mask
        :
        """
        mask = self['x'][:, column_idx]
        return mask


    def slices_by_key(self, attribute, value):
        """
        Masks internal values in accordance with the
        Args:
            attribute: (str) attribute to return
            value: mask values that are NOT equal to this parameter
        :
        """
        slices = self._store._slice_dict
        mask = np.array(self.wds_names) == value
        start_idx = slices[attribute][:-1][mask]
        end_idx = slices[attribute][1:][mask]
        return start_idx, end_idx

    # @property
    # def positional_indices(self) -> int:
    #     r"""Returns the number of features per node in the graph."""
    #     return None
    #
    # def __getitem__(self, idx: Union[int, np.integer, str, IndexType]) -> Any:
    #     if (isinstance(idx, (int, np.integer))
    #             or (isinstance(idx, Tensor) and idx.dim() == 0)
    #             or (isinstance(idx, np.ndarray) and np.isscalar(idx))):
    #         return self.get_example(idx)
    #     elif isinstance(idx, str) or (isinstance(idx, tuple)
    #                                   and isinstance(idx[0], str)):
    #         # Accessing attributes or node/edge types:
    #         return super().__getitem__(idx)
    #     else:
    #         return self.index_select(idx)


class Data(torch_geometric.data.Data):
    def __init__(self, x: OptTensor = None, y: OptTensor = None, *args, **kwargs):
        super().__init__(x=x, y=y, *args, **kwargs)

    @property
    def wds_names(self) -> Any:
        return self['wds_names'] if 'wds_names' in self._store else None

@profile
def collate(
        cls,
        data_list: List[BaseData],
        increment: bool = True,
        add_batch: bool = True,
        follow_batch: Optional[Union[List[str]]] = None,
        exclude_keys: Optional[Union[List[str]]] = None,
) -> Tuple[BaseData, Mapping, Mapping]:
    # Collates a list of `data` objects into a single object of type `cls`.
    # `collate` can handle both homogeneous and heterogeneous data objects by
    # individually collating all their stores.
    # In addition, `collate` can handle nested data structures such as
    # dictionaries and lists.

    if not isinstance(data_list, (list, tuple)):
        # Materialize `data_list` to keep the `_parent` weakref alive.
        data_list = list(data_list)

    if cls != data_list[0].__class__:
        out = cls(_base_cls=data_list[0].__class__)  # Dynamic inheritance.
    else:
        out = cls()

    # Create empty stores:
    out.stores_as(data_list[0])

    follow_batch = set(follow_batch or [])
    exclude_keys = set(exclude_keys or [])

    # Group all storage objects of every data object in the `data_list` by key,
    # i.e. `key_to_store_list = { key: [store_1, store_2, ...], ... }`:
    key_to_stores = defaultdict(list)
    for data in data_list:
        for store in data.stores:
            key_to_stores[store._key].append(store)

    # With this, we iterate over each list of storage objects and recursively
    # collate all its attributes into a unified representation:

    # We maintain two additional dictionaries:
    # * `slice_dict` stores a compressed index representation of each attribute
    #    and is needed to re-construct individual elements from mini-batches.
    # * `inc_dict` stores how individual elements need to be incremented, e.g.,
    #   `edge_index` is incremented by the cumulated sum of previous elements.
    #   We also need to make use of `inc_dict` when re-constructuing individual
    #   elements as attributes that got incremented need to be decremented
    #   while separating to obtain original values.
    device = None
    slice_dict, inc_dict = defaultdict(dict), defaultdict(dict)
    for out_store in out.stores:
        key = out_store._key
        stores = key_to_stores[key]
        for attr in stores[0].keys():

            if attr in exclude_keys:  # Do not include top-level attribute.
                continue

            values = [store[attr] for store in stores]

            # The `num_nodes` attribute needs special treatment, as we need to
            # sum their values up instead of merging them to a list:
            if attr == 'num_nodes':
                out_store._num_nodes = values
                out_store.num_nodes = sum(values)
                continue

            # Skip batching of `ptr` vectors for now:
            if attr == 'ptr':
                continue

            # Collate attributes into a unified representation:
            value, slices, incs = _collate(attr, values, data_list, stores,
                                           increment)

            if isinstance(value, Tensor) and value.is_cuda:
                device = value.device

            out_store[attr] = value
            if key is not None:
                slice_dict[key][attr] = slices
                inc_dict[key][attr] = incs
            else:
                slice_dict[attr] = slices
                inc_dict[attr] = incs

            # Add an additional batch vector for the given attribute:
            if (attr in follow_batch and isinstance(slices, Tensor)
                    and slices.dim() == 1):
                repeats = slices[1:] - slices[:-1]
                batch = repeat_interleave(repeats.tolist(), device=device)
                out_store[f'{attr}_batch'] = batch

        if type(data_list[0]) == GraphData:
            element = 'node'
        else:
            element = 'edge'

        # In case the storage holds node, we add a top-level batch vector it:
        if (add_batch and isinstance(stores[0], NodeStorage)
                and stores[0].can_infer_num_nodes):
            repeats = [getattr(store, f'num_{element}s') for store in stores]
            out_store.batch = repeat_interleave(repeats, device=device)
            out_store.ptr = cumsum(torch.tensor(repeats, device=device))

    return out, slice_dict, inc_dict
