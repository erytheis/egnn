from abc import abstractmethod
from os import path as osp
from typing import List

import networkx as nx
import pandas as pd
import torch

# from src.surrogate_models.torch_models.data.complex_data import Cochain
from src.surrogate_models.torch_models.parse_config import load_test_config
from src.utils.utils import all_equal
import numpy as np


class BaseGNNDataset:
    """
    WIP
    Base class NN models
    """

    def __add__(self, other):
        pass

    @property
    def datasets(self):
        return [self]

    @classmethod
    def make_new(cls):
        self = cls.__new__(cls)
        self.data = None
        self.slices = None
        # self.
        # etc etc, then
        return self

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed_gnn')

    def data_to(self, device):
        raise NotImplementedError

    @abstractmethod
    def node_in_channels(self):
        raise NotImplementedError

    @abstractmethod
    def edge_attr_in_channels(self):
        raise NotImplementedError

    def to_networkx(self, weight_idx=None):
        graphs = []
        for g in self:
            graph = torch_geometric.utils.to_networkx(g)
            for i, (u, v) in enumerate(g.edge_index.t().tolist()):
                graph[u][v]['weight'] = g.edge_attr[i, weight_idx]
            graph = nx.Graph(graph)
            graphs.append(graph)

    @property
    def num_train_subsets(self):
        return 1

def read_file(folder, prefix, name, dtype=None):
    path = osp.join(folder, '{}{}.csv'.format(prefix, name))
    return read_txt_array(path, sep=',', dtype=dtype)


def parse_txt_array(src, sep=None, start=0, end=None, dtype=None, device=None):
    to_number = int
    if torch.is_floating_point(torch.empty(0, dtype=dtype)):
        to_number = float

    src = [[to_number(x) for x in line.split(sep)[start:end]] for line in src]
    src = torch.tensor(src).to(dtype).squeeze()
    return src


def parse_csv_array(src):
    A = pd.read_csv(src, index_col=0)
    # A.T.apply(lambda x: pd.factorize(x)[0]).to_numpy()
    edge_signals = pd.read_csv(folder + '/edge_signals.csv', index_col=0)


def read_txt_array(path, sep=None, start=0, end=None, dtype=None, device=None):
    with open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return parse_txt_array(src, sep, start, end, dtype, device)


def split(data, node_signal_indicators, edge_signal_indicators):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(node_signal_indicators)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(edge_signal_indicators)), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    pass


def insert_properties(signal, properties, indicators):
    pass


def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    else:
        # Imitate `collate` functionality:
        data._num_nodes = torch.bincount(batch).tolist()
        data.num_nodes = batch.numel()
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    return data, slices


def get_slices(indices, number_of_elements):
    if isinstance(number_of_elements, int):
        slices = indices.repeat(number_of_elements).to_numpy()

    slices = np.unique(slices, return_inverse=True)[1]
    slices = torch.cumsum(torch.from_numpy(np.bincount(slices)), 0, dtype=torch.int)
    slices = torch.cat([torch.tensor([0]), slices])
    return slices


def _extract_graph_shapes(df, df_indicators):
    number_of_graphs = int(df.shape[0] / df_indicators.shape[0])
    number_of_elements = df.shape[1]
    return number_of_graphs, number_of_elements


def process_signals(signals_df, indicators_df, skip_features=[], imputer=None, dtype=torch.float32):
    # filter out features
    signals_df = signals_df[~indicators_df['name'].isin(skip_features)]
    indicators_df = indicators_df[~indicators_df['name'].isin(skip_features)]

    signal_names = indicators_df['name'].unique()

    number_of_elements = signals_df.shape[1]
    number_of_graphs = int(indicators_df['idx'].shape[0] / len(signal_names))

    # slices = get_slices(indicators_df['idx'], number_of_elements, signals_df) / len(signal_names)
    signal_counts = signals_df.count(axis=1)
    signal_counts = signal_counts.iloc[::len(signal_names)]
    slices = torch.cumsum(torch.from_numpy(signal_counts.to_numpy()), 0, dtype=torch.int)
    slices = torch.cat([torch.tensor([0]), slices])

    signals = [signals_df[indicators_df['name'] == name] for name in signal_names]

    if imputer is not None:
        for i, signal in enumerate(signals):
            mask = np.isnan(signal)
            masked_df = np.ma.masked_array(signal, mask)
            fill_value = pd.DataFrame({col: signal.mean(axis=1) for col in signal.columns})
            masked_df.filled(fill_value)
            signals[i] = masked_df
        signals = [signal.reshape(-1) for signal in signals]
    else:
        signals = [signal.stack() for signal in signals]
    signals = np.array(signals, dtype=pd.Series).T.astype(float)
    signals = torch.tensor(signals).to(dtype).squeeze()

    return signals, slices.long(), number_of_graphs, signal_names


def merge_datasets(datasets: List[BaseGNNDataset]):
    ds_cls = datasets[0].__class__
    new_ds = ds_cls.make_new()
    # new_ds.data = torch.cat([x for x in ])
    pass


def concat_gnn_datas(datas: List):
    if isinstance(datas[0], dict):
        return datas[0][0].__class__.concat_datas(datas)

    xs, ys, edge_attrs, edge_indices, wds_names = [], [], [], [], []

    for data in datas:
        xs.append(data.x)
        ys.append(data.y)
        edge_attrs.append(data.edge_attr)
        edge_indices.append(data.edge_index)
        wds_names.append(data.wds_names)

    assert all_equal([data.x_names for data in datas])
    assert all_equal([data.y_names for data in datas])
    assert all_equal([data.edge_attr_names for data in datas])

    x = torch.cat(xs, axis=0)
    y = torch.cat(ys, axis=0)
    edge_attr = torch.cat(edge_attrs, axis=0)
    edge_index = torch.cat(edge_indices, axis=1)

    new_data = type(datas[0])(
        x=x,
        y=y,
        edge_index=edge_index,
        edge_attr=edge_attr,
        x_names=data.x_names,
        y_names=data.y_names,
        edge_attr_names=data.edge_attr_names,
        wds_names=wds_names)
    # torch.cat(, axis=0).shape
    return [new_data]


gnn_file_names = [
    'A', 'node_labels', 'node_properties', 'node_signals',
    'edge_labels', 'edge_attributes', 'edge_signals', 'graph_labels', 'graph_attributes'
]

if __name__ == '__main__':
    from src.surrogate_models.torch_models.dataset.transforms.transforms import *

    config = load_test_config()
    gnn_ds = load_datasets(config, device=torch.device('cpu'))
    print(config)
