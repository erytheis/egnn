import warnings
from typing import Optional

import cv2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wntr
from numpy import ndarray
from torch import Tensor
from torch_geometric.data import Data

from src.surrogate_models.torch_models.model.metric import mae_per_graph
from src.utils.wds_utils import plot_network, plot_network_3d



def plot_results(model):
    model.forward()


def plot_prediction_comparison_with_demands(predictions: ndarray,  # (BxN) B - nubmer of graphs, N - nodes per graph
                                            true_labels: ndarray,  # (BxN)
                                            secondary_values: ndarray,
                                            secondary_label: Optional[str] = '',
                                            xticks=None,
                                            ax2=None,
                                            c='paleturquoise',
                                            order=True,
                                            *args, **kwargs):
    ax = plot_prediction_comparison(predictions, true_labels, xticks, order, *args,
                                    **kwargs)

    if order:
        sort_args = true_labels.argsort()
        secondary_values = secondary_values[sort_args]
    if ax2 is None:
        ax2 = ax.twinx()

    ax2.plot(secondary_values, marker='o', c=c,
             markersize=3,
             zorder=2,
             linestyle='None',
             alpha=0.7, label=secondary_label)

    return ax


def get_output_by_index(batch: Data,
                        output: Tensor,
                        y_size: Optional[int] = None, idx=0, key='x', label_key='y'):
    """
    Tiles batch graphs into a (1-Y) array,
    where Y is the dim of Y

    :param output: predictions
    :param batch:
    :return:
    """
    size_n = 1
    with torch.no_grad():
        if y_size is None:
            y_size = batch[0].y.detach().numpy().shape[0]

        output = output.detach().numpy().squeeze()

        predictions = np.zeros((size_n, y_size))
        true = np.zeros((size_n, y_size))

        # if self.task == DownstreamTaskTypes.NODE_REGRESSION:
        indices = 0
        graph_slice = batch.slices[key][:size_n + 1]

        labels = batch[label_key].cpu().detach().numpy()[graph_slice[idx + 0]:graph_slice[idx + 1]]

        true[:, :] = labels.squeeze()
        predictions[:, :] = [output[graph_slice[i]:graph_slice[i + 1]] for i in range(size_n)]
    return predictions, true


def plot_prediction_comparison(
        predictions: ndarray,  # (BxN) B - nubmer of graphs, N - nodes per graph
        true_labels: ndarray,
        # (BxN)
        xticks=None,
        order=True,
        ax=None,
        show=False,
        key='Node',
        c_true='orange',
        c_pred='darkcyan',
        value_label='Head',
        prediction_label='Prediction',
        true_label='True',
        bar_kwargs={},
        **scatter_kwargs,
):
    # default scatter_kwargs
    scatter_kwargs.setdefault('s', 80)
    scatter_kwargs.setdefault('alpha', 0.9)
    scatter_kwargs.setdefault('zorder', 2)


    if ax is None:
        fig = plt.figure(facecolor='w', edgecolor='k', figsize=(7, 2))
        ax = plt.gca()

    predictions = predictions[0] if predictions.ndim == 2 else predictions
    true_labels = true_labels[0] if true_labels.ndim == 2 else true_labels

    if order:
        sort_args = true_labels.argsort()
        predictions = predictions[sort_args]
        true_labels = true_labels[sort_args]

    if xticks is None:
        xticks = np.indices(predictions.shape)[0].squeeze()

    ax.set_ylim((true_labels.min(), true_labels.max()))

    ax.bar(xticks, predictions, color=c_pred, label=prediction_label, **bar_kwargs)
    ax.scatter(xticks, true_labels, c=c_true, marker='_',
                label=true_label,  **scatter_kwargs)

    ax.set_xlabel(f'{key} ID, ordered by true value')
    ax.set_ylabel(f'{value_label}')

    # plt.ylim(115, 120)
    # xticks = xticks[::3]

    # plt.xticks(range(len(xticks)), xticks, rotation='vertical')
    ax.autoscale()
    fig.tight_layout() if hasattr(globals(), 'fig') else None

    if show:
        ax.show()
    return ax


def get_worst_prediction_in_batch(batch: Data,
                                  output: Tensor,
                                  order_by=mae_per_graph):
    """
    Gets worst performing graph in a batch into a (1-Y) array, where Y is the dim of Y.


    :param order_by: metric to calculate score per graph
    :param output: predictions
    :param batch:
    :return:
    """

    with torch.no_grad():
        scores = order_by(output.detach().numpy(),
                          batch.y.detach().numpy(),
                          batch_indices=batch.batch.detach().numpy(),
                          ptr=batch.ptr.detach().numpy())
        idx = scores.argmax()
        y = batch[idx].y.detach().numpy()
        output = output.detach().numpy().squeeze()
        predictions = output[batch.ptr[idx]:batch.ptr[idx + 1]]

        true = np.expand_dims(y, axis=0)
        predictions = np.expand_dims(predictions, axis=0)
    return predictions, true, idx


class WDSPlotter:
    """
    Plotting module for a training
    """

    # plt.set_loglevel('WARNING')

    def __init__(self,
                 datasets=None):
        self.datasets = datasets
        # load wns for plotting
        self.wns = {}
        for dataset in datasets.datasets:
            self.wns[dataset.name] = wntr.network.WaterNetworkModel(dataset.inp_filename)

    def tile_inputs(self, batch: Data, size_n: int):
        with torch.no_grad():
            x = batch[0].x.detach().numpy()

            graph_slice = batch.ptr[:size_n + 1]
            tiled = self._tile(batch.x.detach().numpy(), graph_slice, size_n, x.shape[0])
        return tiled

    def tile_edge_attributs(self, batch: Data, size_n: int):
        with torch.no_grad():
            x = batch[0].edge_attr.detach().numpy()

            graph_slice = batch.ptr[:size_n + 1]
            tiled = self._tile(batch.edge_attr.detah().numpy(), graph_slice, size_n, x.shape[0])
        return tiled

    def _tile(self, array, slice, size_n, num_elements):
        tiled = np.zeros((size_n, num_elements))
        tiled[:, :] = [array[slice[i]:slice[i + 1]] for i in range(size_n)]
        return tiled

    def plot_network(
            self,
            node_attributes=None,
            name=None,
            edge_attributes=None,
            edge_index=None,  # add to restore order of the edges
            num_virtual_nodes=0,
            pos=None,
            isometry=False,
            *args,
            **kwargs
    ):
        node_names = self.wns[name].wn.junction_name_list + self.wns[name].wn.tank_name_list + self.wns[
            name].wn.reservoir_name_list
        node_names_dict = dict(zip(node_names, np.arange(len(node_names))))

        # get number of virtual nodes
        ds = [ds for ds in self.datasets.datasets if ds.name == name][0]

        if node_attributes is not None:
            if node_attributes.shape[-1] != len(node_names) + num_virtual_nodes:
                return
            nodes = pd.Series(dict(zip(node_names, node_attributes.T)))
        else:
            nodes = None

        if edge_attributes is not None:
            assert edge_index is not None, "edge_index is required"

            edge_names = list(self.wns[name].wn.get_graph().edges.keys())
            edge_names = {tuple(sorted([node_names_dict[e[0]], node_names_dict[e[1]]])): e[2] for e in edge_names}
            #
            edge_attributes = {edge_names[tuple(sorted(edge_index[:, i].cpu().numpy()))]: e
                               for i, e in enumerate(edge_attributes.squeeze())
                               if tuple(sorted(edge_index[:, i].cpu().numpy())) in edge_names}

        if isometry:
            warnings.warn("Node attributes in Isometry might be mixed")
            return plot_network_3d(self.wns[name].wn, nodes, edge_attributes, pos=pos, *args, **kwargs)
        else:
            return plot_network(self.wns[name].wn, nodes, edge_attributes, pos=pos, *args, **kwargs)

    def __str__(self):
        return "wds_plots"


class AttentionVisualizer(WDSPlotter):

    def __init__(self, datasets):
        super().__init__(datasets=datasets)
        self.attentions = []
        self.edge_indices = []

    def store_attentions(self, atts, edge_index=None):
        self.attentions.append(atts)
        self.edge_indices.append(edge_index) if edge_index is not None else None

    def plot_attention_weights(self, wds_name=None, layer_id=0):
        if wds_name is None:
            wds_name = self.batch[0].wds_name
        attention_weights = self.attentions[layer_id]
        ax = super().plot_network(self.batch.y, wds_name, edge_attributes=attention_weights)


class LayerInspectorPlotter:
    """
    A plotter for layer inspection
    """

    def __init__(self, layer_inspector):
        self.layer_inspector = layer_inspector


def load_wns(datasets):
    wns = {}
    for dataset in datasets:
        wns[dataset.name] = wntr.network.WaterNetworkModel(dataset.inp_filename)
    return wns


def tile_predictions_for_network(batch: Data,
                                 output: Tensor,
                                 wds_name: str,
                                 size_n: int = 1):
    with torch.no_grad():
        y = batch.y.cpu()[batch.mask_by_key('y', wds_name)].detach().numpy()
        output = output.cpu()[batch.mask_by_key('y', wds_name)].detach().numpy().squeeze()

        idx = batch.wds_names.index(wds_name)
        wds_size = batch.ptr[idx + 1] - batch.ptr[idx]

        indices = slice(0, size_n)

        true = y.reshape(-1, wds_size)
        predictions = output.reshape(-1, wds_size)

        errors = predictions - true
    return predictions, true


def pack_graphs(batch, out=None):
    y_s, x_s, graphs_dict, y_preds, ea_s = {}, {}, {}, {}, {}
    graphs = batch[batch.batch.unique()]
    start_idx = 0

    for g in graphs:
        x = g.x.detach().numpy()
        y = g.y.detach().numpy()
        ea = g.edge_attr.detach().numpy()

        # quick fix
        wds_name = g.wds_names if isinstance(g.wds_names, str) else g.wds_names[0]


        if wds_name not in x_s.keys():
            x_s[wds_name] = []
            y_s[wds_name] = []
            graphs_dict[wds_name] = []
            ea_s[wds_name] = []

        if out is not None:
            end_idx = start_idx + g.num_nodes

            y_pred = out[start_idx:end_idx].detach().numpy()
            start_idx = end_idx
            if wds_name not in y_preds.keys():
                y_preds[wds_name] = []
            y_preds[wds_name].append(y_pred)

        x_s[wds_name].append(x)
        y_s[wds_name].append(y)
        graphs_dict[wds_name].append(g)
        ea_s[wds_name].append(ea)
    return x_s, y_s, y_preds, graphs_dict, ea_s


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("conv" in n) and ("weight" in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    ax = plt.gca()
    return ax


def visualize_node_ranges(y, slices):
    for i in range(len(slices)):
        plt.plot(y[slices[i]])
        # edge_attr = d.data['edge_attr'].detach().numpy() if hasattr(d.data,'edge_attr') is not None else None
        # edge_labels = d.data['edge_labels'].detach().numpy() if hasattr(d.data, 'edge_labels') is not None else None

        plt.show()


def align_edge_index_with_networkx(G, node_names_dict,
                                   edge_index: Tensor,
                                   edge_attributes):
    edge_names = list(G.edges.keys())
    edge_names = {tuple(sorted([node_names_dict[e[0]], node_names_dict[e[1]]])): e[2] for e in edge_names}
    #
    edge_attributes = {edge_names[tuple(sorted(edge_index[:, i]))]: e
                       for i, e in enumerate(edge_attributes)
                       if tuple(sorted(edge_index[:, i])) in edge_names}
    return edge_attributes


if __name__ == '__main__':
    checkpoint_path = '/saved/training_logs/models/GNCA/0131_103142/checkpoint-epoch50-loss0.1023.pth'


@torch.no_grad()
def plot_evolution_of_features(features, vmax=None, vmin=None):
    # get a squared shape
    w = int(np.ceil(np.sqrt(len(features))))
    h = int(np.ceil(len(features) / w))

    # set figsize proprtionate to the shape of an array
    fig, axs = plt.subplots(w, h, figsize =(10,10))

    if vmax is None:
        vmax = max([i.max() for i in features.values()])
    if vmin is None:
        vmin = min([i.min() for i in features.values()])

    for i in range(0, w * h):
        ax = axs.flatten()[i]
        img = features[i]
        ax.imshow(img.detach().cpu().numpy(), vmax=vmax, vmin=vmin)
        ax.set_axis_off()
        ax.set_aspect('auto')

        if i + 1 not in features.keys():
            break

    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    return fig