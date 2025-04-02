import numpy as np


import src.surrogate_models
from src.surrogate_models.torch_models.data.simplex import get_boundary_and_laplacian_new, SimplexData
from src.surrogate_models.torch_models.dataset.transforms.base import BaseTransform
from src.utils import torch


class ToData(BaseTransform):

    def forward(self, data, *args, **kwargs):
        x = data.x
        num_graphs = 1
        y = data.y.view(-1, x.shape[0] * num_graphs)
        x = x.view(-1, x.shape[1] * x.shape[0] * num_graphs)
        return src.surrogate_models.torch_models.data.data.Data(x=x, y=y,
                                                                node_names=self.node_names,
                                                                x_names=self.x_names,
                                                                y_names=self.y_names,
                                                                wds_names=self.wds_names
                                                                )

    def _infer_parameters(self, data, *args, **kwargs):
        self.node_names = data.node_names,
        self.x_names = data.x_names,
        self.y_names = data.y_names,
        self.wds_names = data.wds_names


class ToSimplexData(BaseTransform):

    def __init__(self, edge_label='flowrate', node_label=None, normalized=False,
                 remove_self_loops=False, iterative_smoothing_coefficient=0.5,
                 release_ends_of_virtual_edges=False,
                 **kwargs):
        self.edge_label = edge_label
        self.node_label = node_label
        self.normalized = normalized
        self.remove_self_loops = remove_self_loops
        self.release_ends_of_virtual_edges = release_ends_of_virtual_edges
        self.init_kwargs = kwargs
        self.iterative_smoothing_coefficient = iterative_smoothing_coefficient
        super().__init__()

    
    def forward(self, data, *args, **kwargs):
        B_i, B_w, L_i, L_w = get_boundary_and_laplacian_new(data,
                                                            self.normalized,
                                                            self.remove_self_loops,
                                                            release_ends_of_virtual_edges =self.release_ends_of_virtual_edges,
                                                            iterative_smoothing_coefficient=self.iterative_smoothing_coefficient,
                                                            weight_idx=None,
                                                            device=data.x.device)


        if self.edge_label is not None:
            kwargs['edge_y'] = data.edge_attr[:, self.edge_label_index].clone()
            y = kwargs['edge_y']

        if self.node_label is not None:
            kwargs['node_y'] = data.x[:, self.node_label_index].clone()
        else:
            kwargs['node_y'] = data.y.clone()

        return SimplexData(x=data.x,
                           y=data.y,
                           edge_attr=data.edge_attr,
                           edge_index=data.edge_index,
                           wds_names=data.wds_names,
                           laplacian_index=L_i,
                           laplacian_weight=L_w,
                           boundary_index=B_i,
                           boundary_weight=B_w,
                           **self.init_kwargs,
                           **kwargs,
                           )

    def _infer_parameters(self, data, *args, **kwargs):
        self.edge_label_index = data.edge_attr_names.index(self.edge_label)
        if self.node_label is not None:
            self.node_label_index = data.x_names.index(self.node_label)

