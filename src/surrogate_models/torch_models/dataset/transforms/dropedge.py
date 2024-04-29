from typing import Tuple

import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.utils import dropout_edge

from src.surrogate_models.torch_models.data.data import Data
from src.surrogate_models.torch_models.dataset.transforms import BaseTransform
from src.utils.utils import num_connected_components


class DropEdge(BaseTransform):

    def __init__(self, p=0.1,
                 force_undirected=False, training=True,
                 threshold=0.005,
                 attribute_key='x', treshold_key='head',
                 *args, **kwargs):
        super().__init__()
        self.attribute_key = attribute_key
        self.threshold_key = treshold_key
        self.force_undirected = force_undirected
        self.threshold = threshold
        self.p = p
        self.training = training

    def forward(self, data, *args, **kwargs):
        r"""Randomly drops edges from the adjacency matrix
        :obj:`edge_index` with probability :obj:`p` using samples from
        a Bernoulli distribution.

        The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
        or index indicating which edges were retained, depending on the argument
        :obj:`force_undirected`.

        Args:
            edge_index (LongTensor): The edge indices.
            p (float, optional): Dropout probability. (default: :obj:`0.5`)
            force_undirected (bool, optional): If set to :obj:`True`, will either
                drop or keep both edges of an undirected edge.
                (default: :obj:`False`)
            training (bool, optional): If set to :obj:`False`, this operation is a
                no-op. (default: :obj:`True`)

        :rtype: (:class:`LongTensor`, :class:`BoolTensor` or :class:`LongTensor`)

        Examples:

            >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
            ...                            [1, 0, 2, 1, 3, 2]])
            >>> edge_index, edge_mask = dropout_edge(edge_index)
            >>> edge_index
            tensor([[0, 1, 2, 2],
                    [1, 2, 1, 3]])
            >>> edge_mask # masks indicating which edges are retained
            tensor([ True, False,  True,  True,  True, False])

            >>> edge_index, edge_id = dropout_edge(edge_index,
            ...                                    force_undirected=True)
            >>> edge_index
            tensor([[0, 1, 2, 1, 2, 3],
                    [1, 2, 3, 0, 1, 2]])
            >>> edge_id # indices indicating which edges are retained
            tensor([0, 2, 4, 0, 2, 4])
        """
        if self.p < 0. or self.p > 1.:
            raise ValueError(f'Dropout probability has to be between 0 and 1 '
                             f'(got {self.p}')

        if not self.training or self.p == 0.0:
            return data

        row, col = data.edge_index

        # get headloss on edges
        attribute = data.x[:, self.attribute_idx]
        head_loss = (attribute[row] - attribute[col]).abs()

        # select 50% of the edges
        num_edges = row.size(0)
        num_edges_to_drop = int(num_edges * 0.5)

        # select 10 edges with the smallest head_loss
        edge_mask = torch.rand(row.size(0), device=data.edge_index.device) > self.p
        edge_mask[head_loss.argsort()[num_edges_to_drop:]] = True
        # edge_mask[head_loss > self.threshold] = True

        if self.force_undirected:
            edge_mask[row > col] = False

        # edge_index = data.edge_index[:, edge_mask]
        row, col, edge_attr = filter_adj(row, col, data.edge_attr, edge_mask)

        edge_index = torch.stack([row, col], dim=0)

        num_components, _ = num_connected_components(edge_index, num_nodes=data.num_nodes)
        if num_components > 1:
            return data
        else:
            data.edge_index = edge_index
            data.edge_attr = edge_attr
            return data


    def infer_parameters(self, data: Data) -> None:
        self.attribute_idx = data[f'{self.attribute_key}_names'].index(self.threshold_key)


def filter_adj(row: Tensor, col: Tensor, edge_attr: OptTensor,
               mask: Tensor) -> Tuple[Tensor, Tensor, OptTensor]:
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]