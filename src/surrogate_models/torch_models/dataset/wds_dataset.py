import copy
import glob
import os
from os import path as osp
from os.path import join

import numpy as np
import pandas as pd
import torch
from line_profiler_pycharm import profile
from torch import Tensor
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.dataset import IndexType
from torch_geometric.io.tu import cat

from src.surrogate_models.torch_models.data.data import GraphData
from typing import Optional, Callable, List, Union, Mapping

from src.surrogate_models.torch_models.dataset.base_gnn_dataset import BaseGNNDataset, process_signals
from src.utils.utils import get_abs_path, read_json, read_yaml, PROJECT_ROOT
from src.utils.wds_utils import get_hydraulic_resistance


class WDSGNNDataset(InMemoryDataset, BaseGNNDataset):

    def __init__(self, root: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 reload_data: bool = False,
                 skip_features=None,
                 # data loading params
                 normalize_heads: bool = True,
                 negative_heads: bool = True,
                 name=None,
                 positional_features=None,
                 scaling_temperature: Optional[float] = None,
                 cache_transformed: Optional[bool] = False,
                 **kwargs):
        if skip_features is None:
            skip_features = []
        skip_features = ['elevation', 'Open', 'Closed', 'Pipe',
                         'Pump',
                         # 'flowrate',
                         'HydraulicCosts',
                         'CurrentFlowBetweennessCentrality',
                         'InformationCentrality',
                         'InvWeightCurrentFlowBetweennessCentrality'] + skip_features
        self.skip_features = skip_features
        self.name = name
        self.scaling_temperature = scaling_temperature
        self.positional_features = positional_features if positional_features is not None else []
        self.reload_data = reload_data
        self.cache_transformed = cache_transformed

        self.normalize_heads = normalize_heads
        self.negative_heads = negative_heads

        self._transformed: Optional[List[bool]] = None

        super().__init__(get_abs_path(root), transform=transform, pre_transform=pre_transform)

        if reload_data:
            self.process()

        self.data, self.slices = torch.load(self.processed_paths[0])

        self.prepare_transforms()

        # get simulation metadata
        self.inp_filename = None
        if 'config.yaml' in self.list_file_names_in_raw():
            config = read_yaml(join(self.root, 'config.yaml'))
            self.inp_filename = join(PROJECT_ROOT, 'input', 'inp_files', config["fixed_params"]['network'])

    def prepare_transforms(self):

        # workaround with the global strorage of the Data obj
        with torch.no_grad():
            if hasattr(self.pre_transform, 'infer_parameters'):
                self.pre_transform.infer_parameters(self.data)

        # pre_transform the data. We have to do it separately because the edge indices are repeated until
        # collating function is called
        if self.pre_transform is not None:
            print('Pre-Transformed')
            self._data_list = None
            self.data = self.pre_transform(self.data)

        if hasattr(self.transform, 'infer_parameters'):
            self.transform.infer_parameters(self.data)

    @profile
    def __getitem__(
            self,
            idx: Union[int, np.integer, IndexType],
    ) -> Union['Dataset', Data]:
        r"""In case :obj:`idx` is of type integer, will return the data object
        at index :obj:`idx` (and transforms it in case :obj:`transform` is
        present).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type long or
        bool, will return a subset of the dataset at the specified indices."""
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):

            data = self.get(self.indices()[idx])

            if self.transform is None or self._transformed[idx]:
                data = data
            else:
                data = self.transform(data)
            #
            if self.cache_transformed:
                self._data_list[idx] = data
                self._transformed[idx] = True

            return data
        else:
            return self.index_select(idx)

    def clear_cache(self):
        self._transformed = self.len() * [False]
        self._data_list = self.len() * [None]

    @property
    def name(self):
        if self._name is None:
            return os.path.basename(
                os.path.dirname(self.root))
        else:
            return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @profile
    def get(self, idx: int, *args, **kwargs):
        if not hasattr(self, '_transformed') or self._transformed is None:
            self._transformed = self.len() * [False]
        elif self._data_list[idx] is not None and self._transformed[idx]:
            return copy.copy(self._data_list[idx])

        data = super().get(idx, *args, **kwargs)
        # if hasattr(self.data, 'x_names'):
        #     data.x_names = self.data.x_names
        return data

    def get_pre_transformed(self, idx):
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):
            data = super().get(self.indices()[idx])
            return data

    @property
    def processed_dir(self) -> str:
        return BaseGNNDataset.processed_dir.fget(self)

    def process(self):
        data, slices = read_wds_data(self.root,
                                     skip_features=self.skip_features,
                                     name=self.name,
                                     normalize_heads=self.normalize_heads,
                                     negative_heads=self.negative_heads,
                                     )

        self.data, self.slices = data, slices

        self._data_list = None

        torch.save((self.data, slices), self.processed_paths[0])

    @property
    def num_node_attributes(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_attributes(self) -> int:
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    def list_file_names_in_raw(self):
        files = glob.glob(osp.join(self.root, '*'))
        return [f.split(os.sep)[-1] for f in files]

    def data_to(self, device):
        self.data.to(device)

    def concat_keys(self):
        return self.data.keys()

    @property
    def node_in_channels(self):
        return self.data.x.size(1) if self.data.x is not None else 0

    @property
    def edge_attr_in_channels(self):
        return self.data.edge_attr.size(1) if self.data.edge_attr is not None else 0

    def delete_graph(self, key):
        indices = range(len(self))
        indices = [i for i in indices if i != key]
        # new_data = resample_batches(indices, self.data, self.slices)
        new_data = [self.get(i) for i in indices]
        new_data = type(self.data).from_list(new_data)
        new_slices = {}
        for s in self.slices:
            left = self.slices[s][:key]
            right = self.slices[s][key + 1:]
            graph_size = self.slices[s][key] - left[-1]
            new_slices[s] = torch.cat([self.slices[s][:key], right - graph_size])
        # add keys from the original data
        for key in self.data.keys:
            if key not in new_data.keys:
                new_data[key] = self.data[key]
        self.data = new_data
        self.slices = new_slices


def read_wds_data(folder,
                  prefix='',
                  skip_features=None,
                  name=None,
                  normalize_heads=False,
                  negative_heads=False):
    print("Reading folder {}".format(folder))
    files = glob.glob(osp.join(folder, '{}*.csv'.format(prefix)))
    names = [f.split(os.sep)[-1][len('') + 0:-4] for f in files]

    A = pd.read_csv(folder + '/A.csv')
    edge_names = A.columns.values.tolist()

    number_of_graphs = 1
    slices = {}
    node_signals = node_properties = node_labels \
        = edge_signals = edge_properties = edge_labels \
        = node_names = node_property_names \
        = node_signal_names = node_label_names = None
    edge_property_names = edge_signal_names = []

    if 'node_signals' in names:
        node_signals = pd.read_csv(folder + '/node_signals.csv')
        node_names = node_signals.columns.values.tolist()
        node_signal_indicators = pd.read_csv(folder + '/node_signal_indicators.csv', index_col=False)

        # quickfix to subtract head from the maximum head in that network
        if normalize_heads:
            node_signals.loc[node_signal_indicators['name'] == 'head'] = node_signals.loc[
                node_signal_indicators['name'] == 'head'].sub(
                node_signals.loc[node_signal_indicators['name'] == 'head'].max(axis=1), axis=0)
        if negative_heads:
            node_signals.loc[node_signal_indicators['name'] == 'head'] = -node_signals.loc[
                node_signal_indicators['name'] == 'head']

        node_signals, node_slices, number_of_graphs, node_signal_names = process_signals(node_signals,
                                                                                         node_signal_indicators,
                                                                                         skip_features,
                                                                                         imputer=True)
        if 'HydraulicCosts' in node_signal_names:
            hc_idx = list(node_signal_names).index('HydraulicCosts')
            node_signals[:, hc_idx] = torch.log(node_signals[:, hc_idx])
            node_signals[:, hc_idx][node_signals[:, hc_idx] == -np.inf] = -10
        slices['x'] = node_slices

    if 'edge_signals' in names:
        edge_signals = pd.read_csv(folder + '/edge_signals.csv')
        edge_signal_indicators = pd.read_csv(folder + '/edge_signal_indicators.csv', index_col=False)
        edge_signals, edge_slices, number_of_graphs, edge_signal_names = process_signals(edge_signals,
                                                                                         edge_signal_indicators,
                                                                                         skip_features,
                                                                                         imputer=True)
        edge_signal_names = edge_signal_names.tolist()
        slices['edge_attr'] = edge_slices

    if 'node_properties' in names:
        node_properties = pd.read_csv(folder + '/node_properties.csv', index_col=0)
        node_property_names = node_properties.index.to_list()
        node_names = node_properties.columns.values.tolist()
        node_properties = np.tile(node_properties.T.to_numpy(), (number_of_graphs, 1))
        node_properties = torch.tensor(node_properties).to(dtype=torch.float32).squeeze()
        node_properties = node_properties[:, ~np.isin(node_property_names, skip_features)]
        node_property_names = [name for name in node_property_names if name not in skip_features]

    if 'edge_properties' in names:
        edge_properties = pd.read_csv(folder + '/edge_properties.csv', index_col=0)
        edge_property_names = edge_properties.index.to_list()
        edge_properties = np.tile(edge_properties.T.to_numpy(), (number_of_graphs, 1))
        edge_properties = torch.tensor(edge_properties).to(dtype=torch.float32).squeeze()
        edge_properties = edge_properties[:, ~np.isin(edge_property_names, skip_features)]
        edge_property_names = [name for name in edge_property_names if name not in skip_features]

    if 'node_labels' in names:
        node_labels = pd.read_csv(folder + '/node_labels.csv')
        node_names = node_labels.columns.values.tolist()
        node_label_indicators = pd.read_csv(folder + '/node_label_indicators.csv', index_col=False)

        # quickfix to subtract heads from the maximum head in that network
        if normalize_heads:
            node_labels.loc[node_label_indicators['name'] == 'head'] = node_labels.loc[
                node_label_indicators['name'] == 'head'].sub(
                node_labels.loc[node_label_indicators['name'] == 'head'].max(axis=1), axis=0)
        if negative_heads:
            node_labels.loc[node_label_indicators['name'] == 'head'] = -node_labels.loc[
                node_label_indicators['name'] == 'head']

        node_labels, node_slices, number_of_graphs, node_label_names = process_signals(node_labels,
                                                                                       node_label_indicators)
        slices['y'] = node_slices

    if 'edge_labels' in names:
        edge_labels = pd.read_csv(folder + '/edge_labels.csv')
        edge_label_indicators = pd.read_csv(folder + '/edge_label_indicators.csv',
                                            index_col=False)
        edge_labels, edge_slices, number_of_graphs, edge_label_names = process_signals(edge_labels,
                                                                                       edge_label_indicators)

    # subtract lowest elevation point from the heads and ys
    if 'elevation' in node_property_names:
        elevation_idx = node_property_names.index('elevation')
        if 'head' in node_signal_names:
            head_idx = np.where(node_signal_names == 'head')
            node_signals[:, head_idx[0]] = node_signals[:, head_idx[0]] - node_properties[:, elevation_idx].min()
        if 'head' in node_label_names:
            node_labels = node_labels - node_properties[:, elevation_idx].min()

    # get edge indices
    if len(A) == 2:
        drop_edges = None
        edge_index = np.tile(A.to_numpy(), (1, number_of_graphs))
        slices['edge_index'] = torch.tensor(np.arange(0, number_of_graphs + 1) * A.shape[-1], dtype=torch.long)
    else:
        _ = np.repeat(np.arange(A.shape[0] / 2), 2) * A.max().max()
        # A = A.add(_, axis=0)
        edge_index = np.hstack(A.to_numpy().reshape(-1, 2, A.shape[1]))
        drop_edges = np.isnan(edge_index).all(axis=0)
        edge_index = edge_index[:, ~drop_edges]
        edge_slices = np.zeros(number_of_graphs + 1)
        for i, row in enumerate(A.to_numpy().reshape(-1, 2, A.shape[1])):
            edge_slices[i + 1] = edge_slices[i] + (1 - np.isnan(row).all(axis=0)).sum()
        slices['edge_index'] = torch.tensor(edge_slices, dtype=torch.long)
        # edge_properties = edge_properties[~drop_edges] if drop_edges is not None else edge_properties

    edge_index = torch.tensor(edge_index).to(dtype=torch.long).squeeze()

    # combine node attributes
    x = cat([node_signals, node_properties])
    x = x[~torch.any(x.isnan(), dim=1)]
    x_names = node_signal_names.tolist() + node_property_names

    # combine edge attributes
    edge_attributes = cat([edge_signals, edge_properties])
    edge_attributes = edge_attributes[~drop_edges] if drop_edges is not None else edge_attributes
    slices['edge_attr'] = slices['edge_index']
    edge_attr_names = edge_signal_names + edge_property_names

    # get_hazen_williams  headloss  #TODO place it somewhere else
    length = edge_attributes[:, edge_attr_names.index('length')]
    roughness = edge_attributes[:, edge_attr_names.index('roughness')]
    diameter = edge_attributes[:, edge_attr_names.index('diameter')]
    loss_coefficient = get_hydraulic_resistance(length, diameter, roughness)

    if 'flowrate' in edge_attr_names:
        flowrate_ = edge_attributes[:, edge_attr_names.index('flowrate')]
        flowrate_scaled = torch.sign(flowrate_) * (flowrate_.abs() ** 1.852) * loss_coefficient
        if 'flowrate_scaled' not in skip_features:

            edge_attributes = torch.cat([edge_attributes, flowrate_scaled.unsqueeze(-1)], dim=1)
            edge_attr_names = edge_attr_names + ['flowrate_scaled']

        # Run tests
        for i in range(len(slices['y']) - 1):
            # Headloss test
            heads = node_labels[slices['y'][i]:slices['y'][i + 1]]
            from_nodes = edge_index[0, slices['edge_index'][i]:slices['edge_index'][i + 1]]
            to_nodes = edge_index[1, slices['edge_index'][i]:slices['edge_index'][i + 1]]
            hl = heads[to_nodes] - heads[from_nodes]
            fl = flowrate_scaled[slices['edge_attr'][i]:slices['edge_attr'][i + 1]]
            # fl_ = flowrate_[slices['edge_attr'][i]:slices['edge_attr'][i + 1]]
            err = np.abs(hl - fl) if negative_heads else np.abs(hl + fl)

            # catch large discrepancies between headloss from heads and from flowrates
            relative_err = (err / (np.abs(hl) + 1e-2)) > 1e-2
            abs_err = err > 1e-2
            hl_err = np.abs(hl) > 1e-3
            discrepancy = torch.all(torch.stack([abs_err, relative_err, hl_err]), dim=0)

            if torch.any(discrepancy):
                print('Discrepancy found in graph: {}'.format(i))
                print('Headloss: {}'.format(hl[discrepancy]))
                print('Flowrate: {}'.format(fl[discrepancy]))
                print('Error: {}'.format(err[discrepancy]))



    if name is not None:
        slices['wds_names'] = torch.arange(len(slices['y']), dtype=torch.long)
        name = [name] * (len(slices['y']) - 1)

    # construct labels
    y, y_names = None, None

    # run tests
    data = GraphData(x=x,
                     edge_index=edge_index,
                     edge_attr=edge_attributes,
                     y=y,
                     edge_names=edge_names,
                     node_names=node_names,
                     x_names=x_names,
                     y_names=y_names,
                     edge_attr_names=edge_attr_names,
                     wds_names=name
                     )

    return data, slices


def test_discrepancy(hl_from_heads, hl_from_flowrates):
    err = np.abs(hl_from_heads - hl_from_flowrates)

    # catch large discrepancies between headloss from heads and from flowrates
    relative_err = (err / (np.abs(hl_from_heads) + 1e-3)) > 1e-2
    abs_err = err > 1e-2
    hl_err = np.abs(hl_from_heads) > 1e-3
    discrepancy = torch.all(torch.stack([abs_err, relative_err, hl_err]), dim=0)
    return discrepancy
