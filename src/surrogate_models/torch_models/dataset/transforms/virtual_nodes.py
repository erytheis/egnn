from typing import List, Union, Optional

import torch
import torch_geometric
from torch import Tensor
from torch_geometric.data import Data, HeteroData

from src.surrogate_models.torch_models.dataset.transforms.base import BaseTransform


from src.utils.torch.torch_utils import to_undirected


class ToUndirected(torch_geometric.transforms.ToUndirected, BaseTransform):

    def __init__(self, reduce: str = "add", merge: bool = True, coalesce=False,
                 *args, **kwargs):
        BaseTransform.__init__(self, *args, **kwargs)
        self.reduce = reduce
        self.merge = merge
        self.coalesce = coalesce

    def __call__(
            self,
            data: Union[Data, HeteroData],
            *args, **kwargs
    ) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue

            nnz = store.edge_index.size(1)

            if isinstance(data, HeteroData) and (store.is_bipartite()
                                                 or not self.merge):
                src, rel, dst = store._key

                # Just reverse the connectivity and add edge attributes:
                row, col = store.edge_index
                rev_edge_index = torch.stack([col, row], dim=0)

                inv_store = data[dst, f'rev_{rel}', src]
                inv_store.edge_index = rev_edge_index
                for key, value in store.items():
                    if key == 'edge_index':
                        continue
                    if isinstance(value, Tensor) and value.size(0) == nnz:
                        inv_store[key] = value

            else:
                keys, values = [], []
                for key, value in store.items():
                    if key == 'edge_index':
                        continue

                    if store.is_edge_attr(key):
                        keys.append(key)
                        values.append(value)

                store.edge_index, values = to_undirected(
                    store.edge_index, values, reduce=self.reduce, sort_by_row=True, coalesce=self.coalesce)

                for key, value in zip(keys, values):
                    store[key] = value

        return data


class VirtualNode(torch_geometric.transforms.VirtualNode, BaseTransform):
    keys: List = ['x', 'edge_attr']

    def __init__(self, fully_connected=True, undirected=False, keys=None, extend_dimensions=True):
        self.extend_dimensions = extend_dimensions
        self.undirected = undirected
        self.fully_connected = fully_connected
        if keys is not None:
            self.keys = keys
        BaseTransform.__init__(self, )

    
    def forward(self, data):

        r"""Appends a virtual node to the given homogeneous graph that is connected
        to all other nodes, as described in the `"Neural Message Passing for
        Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.
        The virtual node serves as a global scratch space that each node both reads
        from and writes to in every step of message passing.
        This allows information to travel long distances during the propagation
        phase.

        Node and edge features of the virtual node are added as zero-filled input
        features.
        Furthermore, special edge types will be added both for in-coming and
        out-going information to and from the virtual node.
        """

        num_nodes, (row, col) = data.num_nodes, data.edge_index
        edge_type = data.get('edge_type', torch.zeros_like(row))
        num_new_nodes = 1 if self.fully_connected else num_nodes
        num_new_edges = num_nodes if not self.undirected else 2 * num_nodes

        arange = torch.arange(num_nodes, device=row.device)
        full = row.new_full((num_nodes,), num_nodes) if self.fully_connected else arange + num_nodes

        if self.undirected:
            row = torch.cat([row, arange, full], dim=0)
            col = torch.cat([col, full, arange], dim=0)
        else:
            row = torch.cat([row, full], dim=0)
            col = torch.cat([col, arange], dim=0)

        edge_index = torch.stack([row, col], dim=0)

        new_type = edge_type.new_full((num_nodes,), int(edge_type.max()) + 1)
        edge_type = torch.cat([edge_type, new_type, new_type + 1], dim=0)

        for key, value in data.items():
            if key == 'edge_index' or key == 'edge_type':
                continue

            if isinstance(value, Tensor):
                dim = data.__cat_dim__(key, value)
                size = list(value.size())

                fill_value = None
                if key == 'edge_weight':
                    size[dim] = 2 * num_nodes if self.undirected else num_nodes
                    fill_value = 1.
                elif key == 'batch':
                    size[dim] = 1
                    fill_value = int(value[0])
                elif key == 'edge_attr':
                    size[dim] = 2 * num_nodes if self.undirected else num_nodes
                    fill_value = 0.
                elif data.is_node_attr(key):
                    size[dim] = num_new_nodes
                    fill_value = 0.

                if fill_value is not None:
                    new_value = value.new_full(size, fill_value)
                    data[key] = torch.cat([value, new_value], dim=dim)

        data.edge_index = edge_index
        data.edge_type = edge_type

        if 'num_nodes' in data:
            data.num_nodes = data.num_nodes + num_new_nodes

        dim = data.__cat_dim__('y', data.y)
        value = data.y
        size = list(value.size())

        size[dim] = num_new_nodes
        fill_value = 0.

        if fill_value is not None:
            new_value = value.new_full(size, fill_value)
            data.y = torch.cat([value, new_value], dim=dim)

        if len(self.virtual_idx) > 0:
            assert (num_nodes + num_new_nodes) == data.x.shape[0]
            data.edge_attr[-num_new_edges:, self.virtual_idx['edge_attr']] = 1
            data.x[-num_new_nodes:, self.virtual_idx['x']] = 1

        return data

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def infer_parameters(self, data: Data) -> None:
        self.edge_attr_names = data.edge_attr_names
        self.x_names = data.x_names
        self.virtual_idx = {}

        for key in self.keys:
            if self.extend_dimensions:
                # add a new feature x where any value in the row of the mask is true
                new_feature = torch.zeros((data[key].shape[0], 1), dtype=torch.float, device=data[key].device)

                data[key] = torch.cat([data[key], new_feature], dim=1)
                data['{}_names'.format(key)].append('virtual')
                self.virtual_idx[key] = len(getattr(data, f'{key}_names')) - 1


class VirtualSink(VirtualNode):
    """Add a virtual sink to the network. Works the same as virtual node, but adds flows to the edge_attr according to
     water demand
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 mask_reservoir_values=False,
                 *args, **kwargs):
        super(VirtualSink, self).__init__(*args, **kwargs)

        if columns is None:
            columns = ['demand']
        self.columns = columns
        self.mask_reservoir_values = mask_reservoir_values
        self.reservoir_idx = None

    
    def forward(self, data: Data, inverse=False) -> Data:
        # Add a virtual edge from each node to new virtual nodes, adding demand value to a virtual edge attribute
        # Node features are translated to edge features, number of new edges is equal to number of nodes
        if inverse:
            return self.inverse(data)

        num_real_nodes = data.x.shape[0]
        num_real_edges = data.edge_index.shape[1]
        data = super().forward(data)

        # Add a virtual edge attribute from each node to new virtual edge that are obtainable from data.x
        num_virtual_edges = data.num_edges - num_real_edges
        num_edges = data.edge_index.shape[1] - num_virtual_edges

        # add rest of the columns
        for i, c in enumerate(self.columns):
            if c == 'demand':
                continue
            new_feature_idx = self.x_names.index(c)
            x = data.x[:num_real_nodes, new_feature_idx]
            data.edge_attr[num_edges:, self.edge_attr_names.index(c)] = x

        # add demands
        for i, f in enumerate(self.flowrate_indices):
            x = - data.x[:num_real_nodes, self.x_names.index('demand')]
            if i == 1:
                x = x.sign() * x.abs() ** 1.852
                raise NotImplementedError('Fix the flowrate indices in the VirtualSink!')
            data.edge_attr[num_edges:, f] = x

            if self.mask_reservoir_values:
                reservoir_idx = self.x_names.index('Reservoir')
                reservoirs = data.x[:num_real_nodes, reservoir_idx] == 1
                data.edge_attr[-num_real_nodes:][reservoirs, self.x_names.index('demand')] = 0

        # data.edge_attr[:num_edges, self.virtual_idx['edge_attr']] = 1
        #
        # data.x[:num_real_nodes, self.virtual_idx['x']] = 1

        data.num_real_edges = num_real_edges
        data.num_real_nodes = num_real_nodes

        return data

    def infer_parameters(self, data: Data) -> None:
        # add the virtual edge attributes
        for column in self.columns:
            if column == 'demand':
                continue
            new_edge_attr = torch.zeros((data.num_edges, 1), dtype=torch.float, device=data.x.device)
            data['edge_attr'] = torch.cat([data['edge_attr'], new_edge_attr], dim=1)
            data['edge_attr_names'].append(column)

        super().infer_parameters(data)

        self.flowrate_indices = [self.edge_attr_names.index(f) for f in self.edge_attr_names if 'flowrate' in f]

    def inverse(self, data: Data) -> Data:
        # Remove the virtual sink and the virtual edge attributes
        if self.extend_dimensions:
            if hasattr(Data, 'drop_nodes'):
                data.drop_node_by_features(nodes_column_idx=self.virtual_idx['x'],
                                           edges_column_idx=self.virtual_idx['edge_attr'])
            else:
                for key in self.keys:
                    data[key] = data[key][[data[key][:, self.virtual_idx[key]] == 0]]

        return data


class VirtualReservoirNode(VirtualNode):
    """

    Add a virtual node that connects all reservoir nodes
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 *args, **kwargs):
        super(VirtualReservoirNode, self).__init__(*args, **kwargs)

        if columns is None:
            columns = ['demand']
        self.columns = columns
        self.reservoir_idx = None

    def forward(self, data, inverse=False, *args,**kwargs):
        """ add one virtual node and connect it to all reservoir nodes """
        if inverse:
            return self.inverse(data)

        reservoir_idx = self.x_names.index('Reservoir')

        num_new_nodes = 1
        num_nodes = data.num_nodes
        num_reservoirs = int(data.x[:, reservoir_idx].sum().item())
        num_new_edges = num_reservoirs if not self.undirected else 2 * num_reservoirs

        row, col = data.edge_index

        arange = data.x[:, reservoir_idx].nonzero()[:,0]

        if len(arange) == 0:
            return data

        full = torch.full_like(arange, num_nodes)

        if self.undirected:
            row = torch.cat([row, arange, arange], dim=0)
            col = torch.cat([col, full, full], dim=0)
        else:
            row = torch.cat([row, arange], dim=0)
            col = torch.cat([col, full], dim=0)

        edge_index = torch.stack([row, col], dim=0)

        for key, value in data.items():
            if key == 'edge_index' or key == 'edge_type':
                continue

            if isinstance(value, Tensor):
                dim = data.__cat_dim__(key, value)
                size = list(value.size())

                fill_value = None
                if key == 'edge_weight':
                    size[dim] = 2 * num_nodes if self.undirected else num_nodes
                    fill_value = 1.
                elif key == 'batch':
                    size[dim] = 1
                    fill_value = int(value[0])
                elif key == 'edge_attr':
                    size[dim] = num_new_edges
                    fill_value = 0.

                elif data.is_node_attr(key):
                    size[dim] = num_new_nodes
                    fill_value = 0.

                # add virtual index
                attr_names = getattr(self, f'{key}_names', [])
                if 'virtual' in attr_names and not self.extend_dimensions:
                    virtual_idx = attr_names.index('virtual')
                    new_value = value.new_full(size, 0)
                    new_value[:, virtual_idx] = 1
                    data[key] = torch.cat([value, new_value], dim=dim)

                elif fill_value is not None:
                    new_value = value.new_full(size, fill_value)
                    data[key] = torch.cat([value, new_value], dim=dim)

        data.edge_index = edge_index

        if 'num_nodes' in data:
            data.num_nodes = data.num_nodes + num_new_nodes

        dim = data.__cat_dim__('y', data.y)
        value = data.y
        size = list(value.size())

        size[dim] = num_new_nodes
        fill_value = 0.

        if fill_value is not None:
            new_value = value.new_full(size, fill_value)
            data.y = torch.cat([value, new_value], dim=dim)

        if len(self.virtual_idx) > 0:
            assert (num_nodes + num_new_nodes) == data.x.shape[0]
            data.edge_attr[-num_new_edges:, self.virtual_idx['edge_attr']] = 1
            data.x[-num_new_nodes:, self.virtual_idx['x']] = 1

        return data




class VirtualReservoirConnections(VirtualNode):
    """

    Add a virtual node that connects all reservoir nodes
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 *args, **kwargs):
        super(VirtualReservoirConnections, self).__init__(*args, **kwargs)

        if columns is None:
            columns = ['demand']
        self.columns = columns
        self.reservoir_idx = None

    def forward(self, data, inverse=False, *args,**kwargs):
        """ add one virtual node and connect it to all reservoir nodes """
        if inverse:
            return self.inverse(data)

        reservoir_idx = self.x_names.index('Reservoir')

        num_nodes = data.num_nodes
        num_reservoirs = int(data.x[:, reservoir_idx].sum().item())

        # get number of new edges as number of connections between reservoirs
        num_new_edges = num_reservoirs - 1 if not self.undirected else 2 * (num_reservoirs - 1)

        row, col = data.edge_index

        arange = data.x[:, reservoir_idx].nonzero()[:,0]

        if len(arange) == 1:
            return data

        full = torch.full_like(arange, num_nodes)

        if self.undirected:
            raise NotImplementedError
        else:
            row = torch.cat([row, arange[:-1]], dim=0)
            col = torch.cat([col, arange[1:]], dim=0)


        edge_index = torch.stack([row, col], dim=0)

        for key, value in data.items():
            if key == 'edge_index' or key == 'edge_type':
                continue

            if isinstance(value, Tensor):
                dim = data.__cat_dim__(key, value)
                size = list(value.size())

                fill_value = None
                if key == 'edge_weight':
                    size[dim] = 2 * num_nodes if self.undirected else num_nodes
                    fill_value = 1.
                elif key == 'batch':
                    size[dim] = 1
                    fill_value = int(value[0])
                elif key == 'edge_attr':
                    size[dim] = num_new_edges
                    fill_value = 0.

                elif data.is_node_attr(key):
                    continue


                # add virtual index
                attr_names = getattr(self, f'{key}_names', [])
                if 'virtual' in attr_names and not self.extend_dimensions:
                    virtual_idx = attr_names.index('virtual')
                    new_value = value.new_full(size, 0)
                    new_value[:, virtual_idx] = 1
                    data[key] = torch.cat([value, new_value], dim=dim)

                elif fill_value is not None:
                    new_value = value.new_full(size, fill_value)
                    data[key] = torch.cat([value, new_value], dim=dim)

        data.edge_index = edge_index

        if len(self.virtual_idx) > 0:
            assert (num_nodes) == data.x.shape[0]
            data.edge_attr[-num_new_edges:, self.virtual_idx['edge_attr']] = 1
            # data.x[-num_new_nodes:, self.virtual_idx['x']] = 1

        return data

