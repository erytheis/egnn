from abc import abstractmethod

import torch
import torch.nn.modules.activation as activations

import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.nn import LeakyReLU, Dropout, Identity
from torch_geometric.nn import Linear, MessagePassing
from torch_geometric.typing import OptTensor, Adj



class BaseGNN(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    """
    These are functions for the layer inspector. They call the functions of the same name in the super class.
    layer_id is used for logging purposes
    """


    def _act(self, x):
        return self.act(x) if self.act is not None else x

    def _act_out(self,  x):
        return torch.sigmoid(x) if self.act_out else x

    def _lin_out(self, x):
        return self.lin_out(x)

    def _convs(self, layer_id, x, *args, **kwargs):
        if self.convs is None:
            return 0.
        conv = self.convs[layer_id]
        return conv(x, *args, **kwargs)

    def _skip_connection(self, x, x_new):
        if self.alpha is None:
            return x_new
        else:
            x = x + self.alpha * x_new
        return x



def get_mlp_layer(depth, in_channels, hidden_channels, out_channels=None, act=None,
                  weight_initializer='glorot', dropout=0.0, bias= True, embedding_act_out= False,
                  **args):
    if args is None:
        args = {}
    args['weight_initializer'] = weight_initializer

    if act is None:
        act = LeakyReLU(negative_slope=0.2)

    if out_channels is None:
        out_channels = hidden_channels

    layers = []
    for i in range(depth):
        if i == 0:
            layers.append(Linear(in_channels, hidden_channels, **args, bias=bias))
        elif i == depth - 1:
            layers.append(Linear(hidden_channels, out_channels, **args, bias=bias))
        else:
            layers.append(Linear(hidden_channels, hidden_channels, **args, bias=bias))
        if i < depth - 1 or embedding_act_out:
            layers.append(act)
        layers.append(Dropout(dropout))
    module_list = ModuleList(layers)
    return module_list


def get_activation(act):
    if act is None:
        return None
    if act == 'LeakyReLU':
        return LeakyReLU(negative_slope=0.2)
    if act == 'Identity':
        return Identity()
    if isinstance(act, str):
        return getattr(activations, act)()


class ModuleList(torch.nn.ModuleList):

    def forward(self, x):
        for l in self:
            x = l(x)
        return x

    def reset_parameters(self):
        for l in self:
            if hasattr(l, 'reset_parameters'):
                l.reset_parameters()

