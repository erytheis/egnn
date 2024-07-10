from typing import List, Mapping, Any, Optional

import torch
import torch.nn.functional as F
import torch_geometric
from torch import Tensor
from torch.nn import LeakyReLU, ModuleDict
from torch_geometric.nn import GCN2Conv, GCNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptTensor, Adj

from src.surrogate_models.torch_models.model.base_gnn import BaseGNN, get_mlp_layer, get_activation

dim_to_attr = {'node': 'x',
               'edge': 'edge_attr'}
intermediate_dim_to_attr = {'node': 'intermediate_x',
                            'edge': 'intermediate_edge_attr'}

attr_to_dim = {v: k for k, v in dim_to_attr.items()}


class ModuleDict(torch.nn.ModuleDict):
    def forward(self, x: Mapping[Any, Any]):
        x = {dim: module(x) for dim, module in self.items()}
        return x

    def encode(self, x: Mapping[Any, Any], batch=None, *args, **kwargs):
        x = {dim: module.encode(x[dim], batch, *args, **kwargs) for dim, module in self.items()}
        return x

    def process(self, layer_id, x: Mapping[Any, Any], *args, **kwargs):
        x = {dim: module.process(layer_id, x[dim], *args, **kwargs) for dim, module in self.items()}
        return x

    def decode(self, layer_id, x: Mapping[Any, Any], x_init: Mapping[Any, Any], *args, **kwargs):
        x = {dim: module.decode( x[dim], x_init[dim], *args, **kwargs) for dim, module in self.items()}
        return x


class SimplicialCN(torch.nn.Module):

    def __init__(self,
                 num_layers: int,
                 **cochain_kwargs,
                 ):

        super(SimplicialCN, self).__init__()
        self.num_layers = num_layers
        self.cochain_convs = ModuleDict()
        self.converged = False

        for dim, kwargs in cochain_kwargs.items():
            self.cochain_convs[dim] = CochainNetwork(num_layers=num_layers, **kwargs)

    def forward(self, batch):
        x = {'node': batch.x,
             'edge': batch.edge_attr}
        x_input = {dim: x.clone() for dim, x in x.items()}

        x = self.cochain_convs.encode(x)
        x_init = {k: x.clone() for k, x in x.items()}

        self.converged = False

        for layer_id in range(self.num_layers):
            new_x = {}
            #
            # node level message passing
            if 'node' in self.cochain_convs.keys():
                new_x['node'] = self.cochain_convs['node'].process(layer_id, x['node'], index=batch.edge_index,
                                                                   weight=batch.edge_weight,
                                                                   )

            # edge level message passing
            if 'edge' in self.cochain_convs.keys():
                new_x['edge'] = self.cochain_convs['edge'].process(layer_id, x['edge'], index=batch.laplacian_index,
                                                                   weight=batch.laplacian_weight,
                                                                   )

            x = new_x

        x = self.cochain_convs.decode(self.num_layers, x, x_init=x_init)

        x = {dim_to_attr[dim]: x for dim, x in x.items()}

        return x

    def skip_message_passing(self):
        """
        Skip message passing switch
        :return:
        """

        for dim, conv in self.cochain_convs.items():
            for layer in conv.convs:
                if layer.skip_message_passing:
                    layer.skip_message_passing = False
                else:
                    layer.skip_message_passing = True



class CochainNetwork(torch_geometric.nn.models.basic_gnn.BasicGNN, BaseGNN):
    _init_conv_layer_id = 0

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_layers,
                 out_channels=None,
                 dropout=0.0,
                 conv_kwargs=None,
                 alpha=None,
                 # activation parameters
                 act=None,
                 act_out=False,
                 # embedding parameters
                 embedding_layers=1,
                 embedding_bias=False,
                 embedding_act=LeakyReLU(negative_slope=0.2),
                 **kwargs):

        act = get_activation(act)
        embedding_act = get_activation(embedding_act)
        embedding_channels = kwargs.pop('embedding_channels', hidden_channels)


        super().__init__(hidden_channels, hidden_channels,
                         num_layers=num_layers,  act=act,
                         **conv_kwargs)

        # intialize embedding layers
        if in_channels == None:
            self.in_channels = in_channels
        else:
            self.embedding = get_mlp_layer(embedding_layers, in_channels, embedding_channels, hidden_channels,
                                           bias=embedding_bias,
                                           act=embedding_act,
                                           weight_initializer='glorot')


        # initialize output layers
        self.lin_out = Linear(hidden_channels, out_channels, weight_initializer='glorot', bias=False)
        self.act_out = act_out
        self.alpha = alpha

        if self.alpha is None or self.alpha >= 1:
            self.alpha = None

        self.dropout = dropout

    def init_conv(self, in_channels: int, out_channels: int, lin_act=None,
                  **kwargs) -> MessagePassing:

        lin_act = get_activation(lin_act)

        cls_name = kwargs.pop('type', 'SimplicialLayer')

        cls = globals()[cls_name]

        return cls(in_channels, out_channels, normalize=False, act=lin_act, **kwargs)  # .jittable()


    def encode(self, x: Tensor, batch=None, *args, **kwargs) -> Tensor:
        """"""

        xs: List[Tensor] = []

        x = self._embedding(0, x, batch=batch, *args, **kwargs)

        x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def process(self,
                layer_id: int, x, index,  weight=None,
                *args, **kwargs):
        x_new, x_upper, x_lower = 0., 0., 0.

        if layer_id == self.num_layers - 1 and self.jk_mode is None:
            return x

        if not (self.in_channels == None and layer_id == 0):  # skip the first message passing if the input is empty
            x_new = self._convs(layer_id, x, index, weight, *args, **kwargs)
        else:
            x = 0.

        x_new = self._act(x_new)

        x_new = self._skip_connection(x, x_new)

        return x_new

    def decode(self, layer_id, x, x_init=None):
        """
        Decode the latent representation to the ouuput space
        :param x:
        :param x_init: initial embedding
        :return:
        """
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self._lin_out( x)

        return x

    def _embedding(self, x):
        x = self.embedding(x)
        return x




class SimplicialLayer(GCNConv):
    propagate_type = {'x': Tensor, 'edge_weight': Optional[Tensor]}
    skip_message_passing = False

    def __init__(self, in_channels: int, out_channels: int,
                 bias: bool = True, lin_layers=1, act=None, *args, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         bias=bias, *args, **kwargs)

        if lin_layers > 1:  # add more layers to the linear
            self.lin = get_mlp_layer(lin_layers, in_channels, hidden_channels=out_channels,
                                     out_channels=out_channels, bias=False, act=act,
                                     weight_initializer='glorot')
        self.reset_parameters()

    def forward(self, x: Tensor,
                edge_index: Tensor, weights: Tensor,
                size=None
                ):

        x = self.lin(x)
        if self.bias is not None:
            x += self.bias

        if not self.skip_message_passing:
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=weights,
                               size=size,
                               )

        return x

    def __set_size__(self, size: List[Optional[int]], dim: int, src: Tensor):
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.size(self.node_dim)