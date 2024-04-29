import warnings

import numpy as np
from torch_sparse import SparseTensor

import torch
from torch import Tensor
from torch_geometric.data import Data

from src.surrogate_models.torch_models.data.simplex import SimplexData
from src.surrogate_models.torch_models.dataset.transforms import get_propagation_matrix
from src.surrogate_models.torch_models.dataset.transforms.base import BaseTransform


class Mask(BaseTransform):
    """
    Masks values of signals for nodes

    """

    def __init__(self, target_key, target_idx=None, mask_value=None,
                 attribute_key='x',
                 reference_value=1,
                 reference_key=None,
                 reference_idx=None,
                 extend_dimensions=False):

        self.target_key = target_key
        self.target_idx = target_idx

        self.mask_value = mask_value
        self.reference_key = reference_key
        self.reference_value = reference_value
        self.reference_idx = reference_idx
        self.attribute_key = attribute_key
        self.extend_dimensions = extend_dimensions
        super().__init__()

    def forward(self, data, **kwargs):
        # if self.reference_idx is None:
        #     return data

        attribute = data[self.attribute_key]
        mask = self.get_mask(data)

        if self.mask_value == 'mean':
            masked_value = torch.mean(attribute[~mask[:, self.target_idx]][:, self.target_idx])
        elif self.mask_value == 'max':
            masked_value = torch.max(attribute[~mask[:, self.target_idx]][:, self.target_idx])
        elif isinstance(self.mask_value, int) or isinstance(self.mask_value, float):
            masked_value = self.mask_value
        else:
            masked_value = 0
        # elif self.mask_value is 'max':

        data[self.attribute_key] = torch.masked_fill(attribute, mask, value=masked_value)
        if hasattr(self, 'not_masked_idx'):
            data[self.attribute_key][:, self.not_masked_idx] = ~mask.any(axis=1)
        return data

    def get_mask(self, data):

        mask = torch.zeros_like(data[self.attribute_key])
        if self.reference_idx is None and self.reference_idx is None:
            mask_ = torch.ones_like(data[self.attribute_key][:, 0])
        else:
            mask_ = torch.where(data[self.attribute_key][:, self.reference_idx] == self.reference_value, 1, 0)
        mask[:, self.target_idx] = mask_
        return mask.to(torch.bool)

    def _infer_parameters(self, data):
        attribute = data[self.attribute_key]

        if not hasattr(data, '{}_names'.format(self.attribute_key)):
            return data

        names = data[f'{self.attribute_key}_names']

        if self.target_idx is None:
            self.target_idx = names.index(self.target_key)

        if self.mask_value is None:
            self.mask_value = 0

        if hasattr(data, 'dim') and getattr(data, 'dim') == 2:
            return data

        if self.reference_key not in names:
            return data

        if self.extend_dimensions:
            mask = self.get_mask(data)
            # add a new feature x where any value in the row of the mask is true
            new_feature = torch.zeros_like(attribute[:,0])
            new_feature[mask] = 1
            data[self.attribute_key] = torch.cat([attribute, new_feature.unsqueeze(-1)], dim=1)
            data['{}_names'.format(self.attribute_key)].append('not masked')
            self.not_masked_idx = len(data['{}_names'.format(self.attribute_key)]) - 1

        if self.reference_idx is None:
            self.reference_idx = names.index(self.reference_key)



        data[f'{self.attribute_key}_names'] = names

        # elif self.mask_value == 'mean'

class MaskJunctionValues(Mask):
    """
    Masks values of signals for Junction nodes
    :param
    """

    def __init__(self, target_key='head', junction_idx=None,
                 target_idx=None, pump_idx=None, mask_value=None,
                 reference_key='Junction', reference_idx=None, *args, **kwargs):
        super().__init__(target_key, target_idx, mask_value, reference_key=reference_key, reference_idx=reference_idx,
                         *args, **kwargs)
        self.target_key = target_key
        self.target_idx = target_idx
        self.junction_idx = junction_idx
        self.pump_idx = pump_idx

    def get_mask(self, data):
        mask = torch.zeros_like(data.x)
        mask_ = torch.where(data.x[:, self.reference_idx] == 1, 1, 0)
        mask[:, self.target_idx] = mask_

        # unmask the pump nodes
        if self.pump_idx is not None:
            pump_edges = data.edge_attr[:, self.pump_idx] == 1
            pump_nodes = data.edge_index[:, pump_edges == 1]
            mask[pump_nodes] = 0
        return mask.to(torch.bool)

    def _infer_parameters(self, data):
        if self.pump_idx is None and hasattr(data, 'edge_attr_names'):
            if 'Pump' in data.edge_attr_names:
                self.pump_idx = data.edge_attr_names.index('Pump')
        super()._infer_parameters(data)


class Sensor(Mask):

    def __init__(self, sensor_indices=None, *args, **kwargs):
        if sensor_indices is None:
            sensor_indices = []
        self.sensor_indices = sensor_indices
        super().__init__(*args, **kwargs)

    def get_mask(self, data):
        mask = super().get_mask(data)

        # unmask sensor
        mask[self.sensor_indices, self.target_idx] = False

        return mask


class RandomSensor(Mask):

    def __init__(self, number_of_sensors=1, ratio =None, selected_sensor_indices=None,*args, **kwargs):
        self.ratio = ratio
        self.number_of_sensors = number_of_sensors
        self.selected_sensor_indices = selected_sensor_indices
        super().__init__(*args, **kwargs)

    def get_mask(self, data):
        mask = super().get_mask(data)
        # unmask sensor
        masked_rows = ((mask[:, self.target_idx] == True).nonzero(as_tuple=True)[0])  #

        if self.ratio is not None:
            number_of_sensors = int(self.ratio * masked_rows.shape[0])
            number_of_sensors = np.max([0.1 *number_of_sensors, 1])
        else:
            number_of_sensors = self.number_of_sensors

        if self.selected_sensor_indices is not None:
            sampled_indices = torch.randperm(len(self.selected_sensor_indices))[:number_of_sensors]
            sampled_indices = [self.selected_sensor_indices[i] for i in sampled_indices]
        else:
            sampled_indices =torch.randint(len(masked_rows), (number_of_sensors,))

        sensor_rows = masked_rows[sampled_indices]
        mask[sensor_rows, self.target_idx] = False

        return mask


class MaskRandomJunctions(Mask):

    def __init__(self, target_key='head', target_idx=None, mask_value=None, reference_key='not masked',
                 reference_idx=None,
                 p=0.5, *args, **kwargs):
        self.p = 1 - p  # we are actually unmasking the nodes
        super().__init__(target_key, target_idx, mask_value, reference_key, reference_idx, *args, **kwargs)

    def _infer_parameters(self, data):
        super()._infer_parameters(data)
        warnings.warn('MaskRandom: Make sure that all wds have the same number of nodes.')
        nodes_per_graph = data.x.shape[0] / len(data.wds_names)
        assert nodes_per_graph.is_integer(), 'Number of nodes per wd should be an integer. Got {}.'.format(
            nodes_per_graph)
        nodes_per_graph = int(nodes_per_graph)
        # fill the value to show
        perm = torch.stack(
            [torch.randperm(nodes_per_graph)[:int(self.p * nodes_per_graph)] for _ in range(len(data.wds_names))])
        x = torch.zeros(len(data.wds_names), nodes_per_graph)
        for i, p in enumerate(perm):
            x[i].index_fill_(0, p, 1)
        data.x[:, self.reference_idx] = x.reshape(-1) + data.x[:, data.x_names.index('Reservoir')]


class FeaturePropagation(MaskJunctionValues):
    def __init__(self, num_iterations: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_iterations = num_iterations

    def forward(self, batch: Data, edge_weight=None) -> Tensor:
        # out is inizialized to 0 for missing values. However, its initialization does not matter for the final
        # value at convergence
        x = batch.x
        edge_index = batch.edge_index
        if hasattr(batch, 'edge_weight'):
            edge_weight = batch.edge_weight
        mask = self.get_mask(batch)
        mask = ~mask

        out = batch.x
        if mask is not None:
            out = torch.zeros_like(x)
            out[mask] = x[mask]

        n_nodes = x.shape[0]
        adj = get_propagation_matrix(out, edge_index, edge_weight, n_nodes)
        adj = adj + torch.transpose(adj, 0, 1)
        for _ in range(self.num_iterations):
            # Diffuse current features
            out = torch.sparse.mm(adj, out)
            # Reset original known features
            out[mask] = x[mask]
        batch.x = out
        return batch


class AddFlowsToReservoirPipes(BaseTransform):

    def forward(self, data: SimplexData, **kwargs) -> SimplexData:
        value_idx = self.demand_idx
        mask_idx = self.junction_idx
        virtual_edges_idx = self.virtual_edges_idx
        flowrate_idx = self.flowrate_idx
        # reservoir_idx = self.reservoir_idx

        # find reservoirs
        # reservoirs = data.x[:, reservoir_idx] == 1
        real_edges = data.edge_attr[:, virtual_edges_idx] == 0

        # filter rows
        index = data.boundary_index
        weight = data.boundary_weight

        # known values
        B1 = SparseTensor(row=index[0],
                          col=index[1],
                          value=weight,
                          sparse_sizes=(data.num_nodes, data.num_edges), is_sorted=False, trust_data=True)

        x = data.x
        out = data.x[:, 0].unsqueeze(-1)
        out[x[:, mask_idx] == 1, value_idx] = 0
        #
        out = B1.t() @ out

        data.edge_attr[real_edges, flowrate_idx] = out[real_edges].squeeze()
        return data

    def _infer_parameters(self, data, *args, **kwargs):
        self.demand_idx = data.x_names.index('demand')
        self.junction_idx = data.x_names.index('Junction')
        self.flowrate_idx = data.edge_attr_names.index('flowrate')
        self.reservoir_idx = data.x_names.index('Reservoir')
        self.virtual_edges_idx = data.edge_attr_names.index('virtual') if 'virtual' in data.edge_attr_names else None
