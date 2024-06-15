from typing import Optional, List, Mapping, Any

import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn.norm as norms
from line_profiler_pycharm import profile
from torch import Tensor
from torch._torch_docs import merge_dicts
from torch.nn import LeakyReLU, ModuleDict
from torch_geometric.nn import GCNConv, GCN2Conv, GATv2Conv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptTensor, Adj
from torch_scatter import scatter

from src.surrogate_models.torch_models.model.base_gnn import BaseGNN, get_mlp_layer, get_activation
from src.surrogate_models.torch_models.visualization.layer_inspector import LayerInspectorMeta, watch_variable

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
        x = {dim: module.decode(layer_id, x[dim], x_init[dim], *args, **kwargs) for dim, module in self.items()}
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
                                                                   # weight=batch.edge_attr[:, 1],
                                                                   upper_x=x.get('edge', None),
                                                                   x_init=x_init.get('node', None),
                                                                   upper_index=batch.boundary_index,
                                                                   upper_weight=batch.boundary_weight,
                                                                   x_input=x_input.get('node', None))

            # edge level message passing
            if 'edge' in self.cochain_convs.keys():
                new_x['edge'] = self.cochain_convs['edge'].process(layer_id, x['edge'], index=batch.laplacian_index,
                                                                   weight=batch.laplacian_weight,
                                                                   lower_x=x.get('node', None),
                                                                   lower_index=batch.boundary_index.flip(0),
                                                                   x_init=x_init.get('edge', None),
                                                                   lower_weight=batch.boundary_weight,
                                                                   x_input=x_input.get('edge', None)
                                                                   )

            x = new_x
            if self.converged:
                break

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

    def inspect(self, *functions_to_plot):
        for dim, conv in self.cochain_convs.items():
            if hasattr(conv, 'layer_inspector'):
                conv.layer_inspector.active = True
                conv.layer_inspector.functions_to_plot = functions_to_plot

    def get_inspector(self, dim):
        return self.cochain_convs[dim].layer_inspector


class CochainNetwork(torch_geometric.nn.models.basic_gnn.BasicGNN, BaseGNN):
    # , metaclass=LayerInspectorMeta):
    _init_conv_layer_id = 0

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_layers,
                 out_channels=None,
                 dropout=0.0,
                 act=None,
                 norm=None,
                 alpha=None,
                 alpha_init=None,  # alpha for initial connection
                 beta=None,
                 act_out=False,
                 act_first=True,  # whether to apply activation before or after skip connection
                 boundary_condition_idx=None,
                 static_layers=None,
                 norm_first=False,
                 bias=False,
                 global_pool=None,
                 embedding_layers=1,
                 embedding_bias=False,
                 embedding_act=LeakyReLU(negative_slope=0.2),
                 lin_out_bias=False,
                 gating=False,
                 reverse_message_passing=False,
                 boundary_message_passing=True,
                 upper_message_passing=False,
                 lower_message_passing=False,
                 skip_embedding=False,
                 aggregation_step=False,
                 p=None,
                 conv_kwargs=None,
                 *args,
                 **kwargs):

        # if act == 'Identity':
        #     act = Identity()
        # elif isinstance(act, str):
        #     act = getattr(activations, act)()
        act = get_activation(act)
        embedding_act = get_activation(embedding_act)

        if norm is not None:
            norm = getattr(norms, norm)(hidden_channels)

        self.boundary_condition_idx = boundary_condition_idx
        self.boundary_message_passing = boundary_message_passing
        self.upper_message_passing = upper_message_passing
        self.lower_message_passing = lower_message_passing
        self.skip_embedding = skip_embedding

        self.static_layers = static_layers
        self.norm_first = norm_first
        embedding_channels = kwargs.pop('embedding_channels', hidden_channels)

        # fix for the old code
        if 'act' in conv_kwargs:
            conv_kwargs['lin_act'] = conv_kwargs.pop('act')

        super().__init__(hidden_channels, hidden_channels,
                         num_layers=num_layers, norm=norm, act=act,
                         **conv_kwargs)

        # intialize embedding layers
        if global_pool is not None:
            self.global_embedding = Linear(in_channels, hidden_channels // 2, weight_initializer='glorot', bias=bias)
            self.global_pool = 'global_{}_pool'.format(global_pool)
            self.embedding = Linear(in_channels, hidden_channels // 2, weight_initializer='glorot', bias=bias)
        elif in_channels == None:
            self.in_channels = in_channels
        else:
            self.embedding = get_mlp_layer(embedding_layers, in_channels, embedding_channels, hidden_channels,
                                           bias=embedding_bias,
                                           act=embedding_act,
                                           weight_initializer='glorot')

        if not self.boundary_message_passing:
            self.convs = None

        # Initialize message passing layers
        if static_layers:
            self.convs = torch.nn.ModuleList()
            self.convs.append(
                self.init_conv(hidden_channels, hidden_channels, **kwargs))

        # reverse message passing
        self.reverse_message_passing = reverse_message_passing
        if reverse_message_passing:
            self.lin_aggr = torch.nn.ModuleList()
            self.lin_aggr.extend([Linear(hidden_channels * 2, hidden_channels, bias=bias) for _ in range(num_layers)])
            self.convs_out = torch.nn.ModuleList()
            self.convs_out.extend(
                [self.init_conv(hidden_channels, hidden_channels, **conv_kwargs,
                                flow='target_to_source')
                 for _ in range(num_layers)])

        if lower_message_passing:
            self.lower_convs = torch.nn.ModuleList()
            self.lower_convs.extend(
                [self.init_conv(hidden_channels, hidden_channels, message_passing_direction='lower',
                                flow='target_to_source',
                                **conv_kwargs) for _ in range(num_layers)])

        if upper_message_passing:
            self.upper_convs = torch.nn.ModuleList()
            self.upper_convs.extend(
                [self.init_conv(hidden_channels, hidden_channels, message_passing_direction='upper',
                                flow='target_to_source',
                                **conv_kwargs) for _ in range(num_layers)])

        if gating:
            self.p = int(p) if p is not None else None
            self.gatings = torch.nn.ModuleList()
            [self.gatings.append(
                self.init_conv(hidden_channels, hidden_channels, **kwargs))
                for _ in range(num_layers)]

        # initialize output layers
        self.lin_out = Linear(hidden_channels, out_channels, weight_initializer='glorot', bias=lin_out_bias)
        self.act_out = act_out
        self.alpha = alpha
        self.alpha_init = alpha_init

        if self.alpha is None or self.alpha >= 1:
            self.alpha = None

        self.beta = beta
        self.act_first = act_first
        self.dropout = dropout

    def init_conv(self, in_channels: int, out_channels: int, lin_act=None,
                  **kwargs) -> MessagePassing:

        lin_act = get_activation(lin_act)

        cls_name = kwargs.pop('type', 'SimplicialLayer')
        if cls_name == 'SimplicialGCN2Layer':
            self._init_conv_layer_id += 1
            kwargs['layer'] = self._init_conv_layer_id
            return SimplicialGCN2Layer(in_channels, normalize=False, **kwargs).jittable()

        else:
            cls = simplicial_layer_factory(cls_name)

        return cls(in_channels, out_channels, normalize=False, act=lin_act, **kwargs)  # .jittable()

        # returnSimplicialLayer(in_channels, out_channels, normalized=False, **kwargs).jittable())

    @profile
    def encode(self, x: Tensor, batch=None, *args, **kwargs) -> Tensor:
        """"""

        xs: List[Tensor] = []

        x = self._embedding(0, x, batch=batch, *args, **kwargs)

        x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    @profile
    def process(self,
                layer_id: int, x, index, batch=None, weight=None,
                upper_x=None, upper_index=None, upper_weight=None,
                lower_x=None, lower_index=None, lower_weight=None,
                x_init=None, x_input=None,
                *args, **kwargs):
        x_new, x_upper, x_lower = 0., 0., 0.

        x = self._boundary_condition(layer_id, x, x_init,x_input=x_input, *args, **kwargs)

        if layer_id == self.num_layers - 1 and self.jk_mode is None:
            return x

        if not (self.in_channels == None and layer_id == 0):  # skip the first message passing if the input is empty
            x_new = self._convs(layer_id, x, index, weight, *args, **kwargs)

            if self.reverse_message_passing:
                x_out = self.convs_out[layer_id](x, index, weight, *args, **kwargs)
                x_new = x_out - x_new
                # x_new = torch.cat([x_new,  x_out], dim=1)
                # x_new = self.lin_aggr[layer_id](x_new)
        else:
            x = 0.

        if upper_x is not None and hasattr(self, 'upper_convs') and not (self.skip_embedding and layer_id == 0):
            size = tuple([i + 1 for i in upper_index.flip(0).max(dim=1)[0]])
            x_upper = self._upper_convs(layer_id, upper_x, upper_index, upper_weight, size=size, )

            # # quick test
            # B = torch.sparse_coo_tensor(upper_index, upper_weight, size=(size[1],size[0]))
            # out = self.upper_convs[layer_id].lin(upper_x)
            # out = B @ out
            # assert torch.allclose(out, x_upper, atol= 1e-5), 'upper layer error in layer {}'.format(layer_id)

        if lower_x is not None and hasattr(self, 'lower_convs') and not (self.skip_embedding and layer_id == 0):
            size = tuple([i + 1 for i in lower_index.flip(0).max(dim=1)[0]])
            x_lower = self._lower_convs(layer_id, lower_x, lower_index, lower_weight, size=size)

            # # quick test
            # B = torch.sparse_coo_tensor(lower_index, lower_weight, size=(size[1],size[0]))
            # out = self.lower_convs[layer_id].lin(lower_x)
            # out = B @ out
            # assert torch.allclose(out, x_lower, atol= 1e-5), 'lower layer error in layer {}'.format(layer_id)

        if torch.is_tensor(x_upper) or torch.is_tensor(x_lower):
            x_new = x_new + x_upper + x_lower

        # if self.aggregation_step:
        #     x_new = torch.cat([x_new, x, x_init], dim=1)
        #     x_new = self.lin_aggr[layer_id](x_new)

        x_new = self._norms(layer_id, x_new, batch=batch)

        x_new = self._act(layer_id, x_new) if self.act_first else x_new

        x, gating = self._gating(layer_id, x, index, weight, x_init, *args, **kwargs)

        x_new = self._skip_connection(layer_id, x, x_new, gating, x_init)

        x_new = self._act(layer_id, x_new) if not self.act_first else x_new

        return x_new

    # @profile
    def decode(self, layer_id, x, x_init=None):
        """
        Decode the latent representation to the ouuput space
        :param x:
        :param x_init: initial embedding
        :return:
        """
        x = self._beta(self.num_layers, x, x_init) if self.beta is not None else x

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self._lin_out(self.num_layers, x)

        x = self._act_out(self.num_layers, x)
        return x

    def _embedding(self, layer_id, x, batch):
        x_ = self.embedding(x)
        if hasattr(self, 'global_pool') and batch is not None:
            x_glob = self.global_embedding(x)
            x_glob = getattr(torch_geometric.nn.glob, self.global_pool)(x_glob, batch)

            if batch is None:
                x_glob = x_glob.repeat(x.shape[0], 1)
            else:
                x_glob = x_glob[batch]

            x_ = torch.cat([x_, x_glob], dim=-1)
        return x_

    @watch_variable
    def _gating(self, layer_id, x, edge_index, edge_weight, *args, **kwargs):
        gating = 0.
        if hasattr(self, 'gatings'):
            gating = self.gatings[layer_id](x, edge_index, edge_weight, *args, **kwargs)
            if self.p is not None:
                gating = self.act(gating)
                gating = torch.tanh(
                    scatter((torch.abs(gating[edge_index[0]] - gating[edge_index[1]]) ** self.p).squeeze(-1),
                            edge_index[0], 0, dim_size=gating.size(0), reduce='mean'))
            else:
                gating = torch.tanh(gating)

            x = x * gating
        return x, gating

    @watch_variable
    def _lower_convs(self, layer_id, x, index, weight, size, *args, **kwargs):
        return self.lower_convs[layer_id](x, index, weight, size, *args, **kwargs)

    @watch_variable
    def _upper_convs(self, layer_id, x, index, weight, size, *args, **kwargs):
        return self.upper_convs[layer_id](x, index, weight, size, *args, **kwargs)

    def _boundary_condition(self, layer_id, x, x_init, x_input, *args, **kwargs):
        if self.boundary_condition_idx is not None:
            boundary_cells = x_input[:, self.boundary_condition_idx] == 1
            boundary_cells = boundary_cells.unsqueeze(-1).repeat(1, self.hidden_channels)
            x = torch.where(boundary_cells, x_init, x,)
        return x


def simplicial_layer_factory(cls_name):
    """
    Returns a convolutional layer from this module according to the name
    """
    cls = globals()[cls_name]
    return cls





class SimplicialGCN2Layer(GCN2Conv):

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, x_0: OptTensor = None) -> Tensor:
        """"""

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        x.mul_(1 - self.alpha)
        x_0 = self.alpha * x_0[:x.size(0)]

        if self.weight2 is None:
            out = x.add_(x_0)
            out = torch.addmm(out, out, self.weight1, beta=1. - self.beta,
                              alpha=self.beta)
        else:
            out = torch.addmm(x, x, self.weight1, beta=1. - self.beta,
                              alpha=self.beta)
            out = out + torch.addmm(x_0, x_0, self.weight2,
                                    beta=1. - self.beta, alpha=self.beta)

        return out


