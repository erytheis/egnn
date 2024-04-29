import time
from typing import Any, Tuple
from typing import List, Optional

import numpy as np
import scipy
import torch
import torch_sparse
from hodgelaplacians import HodgeLaplacians
from line_profiler_pycharm import profile
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData
from torch_geometric.typing import OptTensor

from src.surrogate_models.torch_models.data.data import GraphData, collate
from src.utils.torch_utils import sparse_eye


class SimplexData(GraphData):
    """
    Extends the torch_geometric.data.Data class to include a simplex attribute.
    At the moment is only limited to  2-simplex.
    """

    def __init__(self, x: OptTensor = None,
                 edge_index: OptTensor = None,
                 edge_attr: OptTensor = None,
                 y: OptTensor = None,
                 pos: OptTensor = None,
                 laplacian_weight: OptTensor = None,
                 laplacian_index: OptTensor = None,
                 boundary_weight: OptTensor = None,
                 boundary_index: OptTensor = None,
                 **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)

        setattr(self._store, 'laplacian_weight', laplacian_weight)
        setattr(self._store, 'laplacian_index', laplacian_index)
        setattr(self._store, 'boundary_weight', boundary_weight)
        setattr(self._store, 'boundary_index', boundary_index)
        setattr(self._store, 'edge_y', kwargs.get('edge_y'))
        setattr(self._store, 'node_y', kwargs.get('node_y'))

    @property
    def laplacian_weight(self):
        return self['laplacian_weight'] if 'laplacian_weight' in self._store else None

    @property
    def laplacian_index(self):
        return self['laplacian_index'] if 'laplacian_index' in self._store else None

    @property
    def boundary_weight(self):
        return self['boundary_weight'] if 'boundary_weight' in self._store else None

    @property
    def boundary_index(self):
        return self['boundary_index'] if 'boundary_index' in self._store else None

    @property
    def edge_y(self):
        return self['edge_y']

    @property
    def node_y(self):
        return self['node_y']

    @profile
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'batch' in key:
            return int(value.max()) + 1
        elif 'laplacian_index' in key:
            return self.num_edges
        elif key == 'boundary_index':
            boundary_inc = self.num_nodes
            cell_inc = self.num_edges
            inc = [[boundary_inc], [cell_inc]]
            return inc
        # elif 'boundary_index' in key:
        #     return self.num_edges
        elif 'index' in key or 'face' in key:
            return self.num_nodes
        else:
            return 0

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

    # def num_nodes(self) -> Optional[int]:
    #     return super().num_edges
    def mask_by_key(self, attribute, value, ):
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

    def mask_by_features(self, attribute='edge_attr', column_idx=-1, value=0):
        """
        Masks internal values in accordance with the
        Args:
            attribute: (str) attribute to return
            column_idx: (int) column index of the feature to mask
        :
        """
        mask = self[attribute][:, column_idx] == value
        return mask

    @profile
    def drop_node_by_features(self, nodes_column_idx=-1, nodes_value=0, edges_column_idx=-1, edges_value=0):
        """
        Drops node corresponding to the value in the x and edge_attr. Drops corresponding edge_index, edge_attr,
        laplacian_index and weights, boundary_index and weights.
        Args:
            nodes_column_idx: (int) column index of the feature to mask
            nodes_value: (int) value to mask
            edges_column_idx: (int) column index of the feature to mask
            edges_value: (int) value to mask
        """
        num_edges = len(self['edge_index'][0])
        num_nodes = len(self['x'])

        nodes_to_keep = self['x'][:, nodes_column_idx] == nodes_value
        edges_to_keep = self['edge_attr'][:, edges_column_idx] == edges_value

        self['x'] = self['x'][nodes_to_keep]
        self['edge_attr'] = self['edge_attr'][edges_to_keep]
        self['edge_index'] = self['edge_index'][:, edges_to_keep]
        if 'node_y' in self:
            self['node_y'] = self['node_y'][nodes_to_keep]
        if 'edge_y' in self:
            self['edge_y'] = self['edge_y'][edges_to_keep]

            # index sparse tensors
        L = torch.sparse_coo_tensor(self.laplacian_index, self.laplacian_weight, (num_edges, num_edges))
        L = L.to_dense()
        L = L[edges_to_keep][:, edges_to_keep]
        L = L.to_sparse()

        self['laplacian_index'] = L._indices()
        self['laplacian_weight'] = L._values()

        # index boundary operators
        B = torch.sparse_coo_tensor(self.boundary_index, self.boundary_weight,
                                    (num_nodes, num_edges))
        B = B.to_dense()[nodes_to_keep][:, edges_to_keep].to_sparse()

        self['boundary_index'] = B._indices()
        self['boundary_weight'] = B._values()



    def get_adjacency(self, dim: Tuple[int]):
        """
        Get adjacency by given dimensions.
        :param dim:
        :return:

        Example:
        1:
            >>> data = SimplexData(edge_index=torch.tensor([[0, 1, 1, 2],
            ...                                             [1, 0, 2, 1]]))

            >>> data.get_adjacency((0, 0))
            torch.tensor([[0, 1, 1, 2],
            [1, 0, 2, 1]]
        2:
            >>> data = SimplexData(laplacian_index=torch.tensor([[0, 1, 1, 2],
            ...                                                   [1, 0, 2, 1]]))

            >>> data.get_adjacency((1, 1))
            torch.tensor([[0, 1, 1, 2],
            [1, 0, 2, 1]]
        """
        if dim == (0, 0):
            return self['edge_index']
        elif dim == (1, 1):
            return self['laplacian_index']
        elif dim == (0, 1):
            return self['boundary_index']
        elif dim == (1, 0):
            return self['boundary_index'].flip(0)
        else:
            raise ValueError(f'Invalid dim: {dim}')

    def get_weights(self, dim: Tuple[int]):
        """
        Get weights by given dimensions.
        :param dim:
        :return:

        Example:
        1:
            >>> data = SimplexData(edge_weight=torch.tensor([1, 2, 3, 4]))

            >>> data.get_weights((0, 0))
            torch.tensor([1, 2, 3, 4]
        2:
            >>> data = SimplexData(laplacian_weight=torch.tensor([1, 2, 3, 4]))

            >>> data.get_weights((1, 1))
            torch.tensor([1, 2, 3, 4]
        """
        if dim == (0, 0):
            return self['edge_weight']
        elif dim == (1, 1):
            return self['laplacian_weight']
        elif dim == (0, 1):
            return self['boundary_weight']
        elif dim == (1, 0):
            return self['boundary_weight']
        else:
            raise ValueError(f'Invalid dim: {dim}')


def get_boundary_and_laplacian(data: SimplexData, device='cpu'):
    simplices = [(s.tolist(), 0.0) for s in data.edge_index.T]
    simplices += [([n], 0.0) for n in range(data.num_nodes)]

    hl = HodgeLaplacians(simplices, mode='gudhi', maxdimension=1)
    B1 = hl.getBoundaryOperator(1)
    B1 = scipy.sparse.coo_matrix(B1)

    boundary_index = torch.tensor(np.vstack((B1.row, B1.col)), dtype=torch.long, device=device)
    boundary_weight = torch.tensor(B1.data, dtype=torch.float, device=device)

    L1 = hl.getHodgeLaplacian(1)

    L1 = scipy.sparse.coo_matrix(L1)

    laplacian_index = torch.tensor(np.vstack((L1.row, L1.col)), dtype=torch.long, device=device)
    laplacian_weight = torch.tensor(L1.data, dtype=torch.float, device=device)

    return boundary_index, boundary_weight, laplacian_index, laplacian_weight


def get_boundary(data: SimplexData, device='cpu', weight_idx=None):
    ei = data.edge_index

    boundary_src = ei.reshape(1, -1).squeeze()
    boundary_dst = torch.arange(0, ei.shape[1], device=device, dtype=torch.long).repeat(2)
    boundary_index = torch.stack((boundary_src, boundary_dst))

    # add weights
    if weight_idx is None:
        boundary_weight = torch.ones(ei.shape[1] * 2, dtype=torch.float, device=device)
        boundary_weight[:ei.shape[1]] = -1
    else:
        boundary_weight = data.edge_attr[:, weight_idx].abs().repeat(2) ** 0.5
        boundary_weight[:ei.shape[1]] *= -1
    return boundary_index, boundary_weight


@profile
def get_boundary_and_laplacian_new(data: SimplexData, normalized=True, remove_self_loops=False, device='cpu',
                                   iterative_smoothing_coefficient=None,
                                   release_ends_of_virtual_edges=False,
                                   weight_idx=None):
    boundary_index, boundary_weight = get_boundary(data, device, weight_idx)
    #
    B1 = torch.sparse_coo_tensor(boundary_index, boundary_weight, size=(data.num_nodes, data.num_edges))
    # B1 = B1.coalesce()

    L = get_L_first_option(B1, normalized)
    if release_ends_of_virtual_edges:
        virtual_edges = data.edge_attr[:, -1] == 1
        L = (L - sparse_eye(L.shape[0],
                            bool_vector=virtual_edges,
                            device=device))

    if remove_self_loops:
        eye = sparse_eye(L.shape[0], value=2., device=device)
        L = L - eye.coalesce()

    if iterative_smoothing_coefficient is not None and iterative_smoothing_coefficient > 0.0:
        L = (sparse_eye(L.shape[0], device=device) - iterative_smoothing_coefficient * L)


    laplacian_index = L.coalesce().indices()
    laplacian_weight = L.coalesce().values()

    return boundary_index, boundary_weight, laplacian_index, laplacian_weight


@profile
def get_L_first_option(B1, normalized=True, ):
    B1 = B1.to_dense() if hasattr(B1, 'to_dense') else B1

    if normalized:
        D = torch.diag(1 / torch.sum(torch.abs(B1), dim=1))
        B_norm = B1.T @ D
        L = torch.mm(B_norm, B1)
    else:
        L = torch.sparse.mm(B1.T, B1)

    L = L.to_sparse()
    return L


@profile
def get_L_2(B1, ):
    B1 = B1.to_dense()

    D = torch.diag(1 / torch.sum(torch.abs(B1), dim=1))

    L = torch.einsum('ij,jj,jk->ik', [B1.T, D, B1])
    L = L.to_sparse()
    return L


def get_L_2_sparse(B1, ):
    # B1 = B1.to_dense()
    D = torch.sparse.sum(torch.abs(B1), dim=1)
    diagonals_values = torch.reciprocal(D.values())

    L = torch.einsum('ij,j,jk->ik', [B1.T, diagonals_values, B1])
    L = L.to_sparse()
    # create sparse matrix in scipy

    return L


@profile
def get_L_torch_sparse(boundary_index, boundary_weight, data, B1=None):
    if B1 is None:
        B1 = torch.sparse_coo_tensor(boundary_index, boundary_weight, size=(data.num_nodes, data.num_edges))
        B1 = B1.coalesce()

    index_T, values_T = torch_sparse.transpose(boundary_index, boundary_weight, data.num_nodes, data.num_edges)
    index_T, values_T = torch_sparse.coalesce(index_T, values_T, data.num_edges, data.num_nodes, op="add")

    D = torch.sparse.sum(torch.abs(B1), dim=1)
    diagonals_values = torch.reciprocal(D.values())
    diagonals_index = torch.arange(diagonals_values.shape[0], device=D.device).repeat(2, 1)
    diagonals_index, diagonals_values = torch_sparse.coalesce(diagonals_index, diagonals_values,
                                                              diagonals_values.shape[0], diagonals_values.shape[0],
                                                              op="add")

    L_indices, L_values = torch_sparse.spspmm(index_T, values_T, diagonals_index, diagonals_values,
                                              data.num_edges, data.num_nodes, data.num_nodes)

    L = torch_sparse.spspmm(L_indices, L_values, boundary_index, boundary_weight, data.num_edges, data.num_nodes,
                            data.num_edges)
    return L


