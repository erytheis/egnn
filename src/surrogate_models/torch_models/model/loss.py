import warnings
from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.nn.functional as F
from line_profiler_pycharm import profile
from torch import Tensor, nn
from torch_scatter import scatter
from torch_sparse import SparseTensor

from src.surrogate_models.torch_models.dataset.transforms import MinMaxNormalize
from src.utils.torch_utils import IterableCompose
from src.utils.utils import Iterator, RUN_CHECKS


def nll_loss(output, batch):
    target = batch.y.view(output.shape)
    return F.nll_loss(output, target)


def mse_loss(output, batch):
    target = batch.y.view(output.shape)
    return F.mse_loss(output, target)


def l1_loss(output, batch):
    target = batch.y.view(output.shape)
    return F.l1_loss(output, target)


class BaseLoss(torch.nn.Module):

    def __init__(self, weight=1, **kwargs):
        self.no_grad = kwargs.pop('no_grad', False)
        self.weight = weight
        super().__init__()

    def __call__(self, *args, **kwargs):
        if self.no_grad:
            with torch.no_grad():
                out = super().__call__(*args, **kwargs)
        else:
            out = super().__call__(*args, **kwargs)

        out = out * self.weight
        out = out if not self.no_grad else out.detach()
        return out


class MSELoss(BaseLoss):

    def __init__(self, weight=1, mask_idx=None, mask_value=0, suffix=None, *args, **kwargs):
        self.mask_idx = mask_idx
        self.suffix = suffix
        self.mask_value = mask_value
        super(MSELoss, self).__init__(weight, **kwargs)

    def forward(self, output, batch):
        output = output['x'] if 'x' in output else output['edge_attr']

        if 'node_y' in batch.keys:
            target = batch.node_y.view(output.shape)
        else:
            target = batch.y.view(output.shape)

        if self.mask_idx is not None:
            mask = batch.x[:, self.mask_idx]
            output = output[mask == self.mask_value]
            target = target[mask == self.mask_value]

        return F.mse_loss(output, target)

    def __str__(self):
        return '{}MSELoss{}'.format('Masked' if self.mask_idx is not None else '',
                                    '' if self.suffix is None else '_{}'.format(self.suffix))


class NodeMSELoss(BaseLoss):
    def __init__(self, weight=1, virtual_idx=None, *args, **kwargs):
        self.virtual_idx = virtual_idx
        super().__init__(weight, **kwargs)

    def forward(self, output, batch):
        target = batch.node_y.view(output['x'].shape)

        output = output['x'] if 'x' in output else output

        return F.mse_loss(output['x'], target)

    def __str__(self):
        return 'NodeMSELoss'


class EdgeMSELoss(BaseLoss):
    def __init__(self, weight=1, edge_label_index=None, virtual_idx=None, network_average=False, *args, **kwargs):
        self.edge_label_index = edge_label_index
        self.virtual_idx = virtual_idx
        self.network_average = network_average
        super().__init__(weight, **kwargs)

    def forward(self, output, batch):
        if isinstance(output, list) or isinstance(output, tuple):
            output = output[1]
        elif isinstance(output, dict):
            output = output['edge_attr']

        if hasattr(batch, 'cochains'):
            target = batch.cochains[1].y.view(output[1].shape)
        elif 'edge_y' in batch.keys:
            target = batch.edge_y.view(output.shape)
        else:
            target = batch.y.view(output.shape)

        if self.virtual_idx is not None:
            virtual = batch.edge_attr[:, self.virtual_idx]
            output = output[virtual == 0]
            target = target[virtual == 0]
            b = batch.batch[virtual == 0]

        if self.network_average:
            errors = (output - target) ** 2
            out = scatter(errors, b.unsqueeze(-1), dim=0, reduce='sum')
            den = scatter(torch.ones_like(b), b, dim=0,
                          reduce='sum')
            return (out / den.unsqueeze(-1)).mean()

        return F.mse_loss(output, target)

    def __str__(self):
        return '{}EdgeMSELoss'.format('Masked' if self.virtual_idx is not None else '')


class WeightedEdgeMSELoss(EdgeMSELoss):
    def __init__(self, dataset, weight=1, weighting_key=None, epsilon=1.0e-2, network_average=False, **kwargs):
        data = dataset.datasets[0].data
        self.epsilon = epsilon
        self.weight_index = data.edge_attr_names.index(weighting_key)
        self.invert = kwargs.pop('invert', False)
        self.network_average = network_average

        super().__init__(weight, **kwargs)

    def forward(self, output, batch):
        if isinstance(output, list) or isinstance(output, tuple):
            output = output[1]
        elif isinstance(output, dict):
            output = output['edge_attr']

        if hasattr(batch, 'cochains'):
            target = batch.cochains[1].y.view(output[1].shape)
        else:
            target = batch.edge_y.view(output.shape)

        with torch.no_grad():
            weights = batch.edge_attr[:, self.weight_index].clone()
            if self.invert:
                weights = 1 / (weights + self.epsilon)
            weights[weights < self.epsilon] = self.epsilon

        if self.virtual_idx is not None:
            virtual = batch.edge_attr[:, self.virtual_idx]
            output = output[virtual == 0]
            target = target[virtual == 0]
            weights = weights[virtual == 0]
            b = batch.batch[virtual == 0]

        errors = (output - target) ** 2

        if self.network_average:
            out = scatter(errors * weights.unsqueeze(-1), b.unsqueeze(-1), dim=0, reduce='sum')
            den = scatter(torch.ones_like(b) * weights, b, dim=0,
                          reduce='sum')
            return (out / den.unsqueeze(-1)).mean()

        return ((errors * weights.unsqueeze(-1))).mean()

    def __str__(self):
        return '{}EdgeMSELoss'.format('Weighted')


class HeadLoss(BaseLoss):
    def __init__(self, dataset, weight=1, p=2, *args, **kwargs):
        data = dataset.datasets[0].data
        self.heads_index = data.x_names.index('head')
        warnings.warn('Make sure that loss coefficient is inversed after propagation')
        self.p = p

        if 'virtual' in data.edge_attr_names:
            self.virtual_index = data.edge_attr_names.index('virtual')
        else:
            self.virtual_index = None

        if 'loss_coefficient' in data.edge_attr_names:
            warnings.warn(
                'MassConservationLoss is used make sure that pipe parameters before H-W coefficient is calculated')
            self.loss_coefficient_index = data.edge_attr_names.index('loss_coefficient')
        else:
            raise ValueError('loss_coefficient not found in edge_attr_names')

        super().__init__(weight, **kwargs)

    def get_headloss_from_flowrates(self, output, batch):

        # get headloss from flowrates
        output = output + 1e-6
        flowrates = (torch.sign(output) * torch.abs(output) ** 1.852).squeeze()

        # get headloss from model
        # denormalize values

        loss_coefficient = torch.abs(batch.edge_attr[:, self.loss_coefficient_index])
        head_loss = flowrates.squeeze() * loss_coefficient

        return head_loss

    def test_validity(self, batch, target, y_head_diff):
        with torch.no_grad():
            loss_coefficient = torch.abs(batch.edge_attr[:, self.loss_coefficient_index])
            virtual = batch.edge_attr[:, self.virtual_index].bool() if self.virtual_index is not None else 1

            hl = y_head_diff[~virtual]
            fl = (torch.sign(target) * torch.abs(target) ** 1.852).squeeze() * loss_coefficient
            fl = fl[~virtual]

            # get absolute discrepancy
            err = torch.abs(hl - fl)

            # catch relative errors
            relative_err = (err / (torch.abs(hl) + 1e-2)) > 1e-2
            abs_err = err > 1e-2
            hl_err = torch.abs(hl) > 1e-3

            # get all errors
            discrepancy = torch.all(torch.stack([abs_err, relative_err, hl_err]), dim=0)

            assert not torch.any(discrepancy), 'Headloss calculation is not valid'

    def __str__(self):
        return 'HeadLoss'

    def forward(self, output, batch):
        if isinstance(output, list) or isinstance(output, tuple):
            output = output[1]
        elif isinstance(output, dict):
            output = output['edge_attr']

        if hasattr(batch, 'cochains'):
            target = batch.cochains[1].y.view(output[1].shape)
        else:
            target = batch.edge_y.view(output.shape)

        # batch = self.transform(batch, inverse=True)

        head_loss = self.get_headloss_from_flowrates(output, batch)
        y_head_diff = get_headloss_from_nodes(batch, heads=batch.node_y)
        virtual = batch.edge_attr[:, self.virtual_index] == 0 if self.virtual_index is not None else 1

        self.test_validity(batch, target, y_head_diff) #if RUN_CHECKS else None

        return head_loss * virtual, y_head_diff * virtual


class HeadlossDiscrepancy(HeadLoss):
    """
    Compares discrepancy between healdoss calculated from edge_attr and x
    """

    def __init__(self, p=2, target='edge_attr', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        assert target in ['label', 'edge_attr'], 'target must be label or edge_attr'
        self.target = target

    def forward(self, output, batch):

        virtual = batch.edge_attr[:, self.virtual_index] == 0 if self.virtual_index is not None else True

        x = self.inverse_transform_heads(output['x'])
        head_loss_x = get_headloss_from_nodes(batch, x).squeeze()[virtual]
        loss_coefficient = torch.abs(batch.edge_attr[:, self.loss_coefficient_index])[virtual]
        flowrate_x = head_loss_x / loss_coefficient
        flowrate_abs = torch.abs(flowrate_x)
        if torch.any(torch.isnan(flowrate_abs)) or torch.any(torch.isinf(flowrate_abs)) or torch.any(flowrate_abs < 0):
            print('nan')
        flowrate_x = (flowrate_x.abs() + 0.000001) ** (1 / 1.852) * torch.sign(flowrate_x)

        # head_loss_edge_attr = self.get_headloss_from_flowrates(output['edge_attr'], batch).squeeze()

        # get headloss from model

        # test validity
        # x_ = self.inverse_transform_heads(batch.node_y)
        # head_loss_x_ = get_headloss_from_nodes(batch, x_)
        # flowrate_x_ = (head_loss_x_ / loss_coefficient)
        # flowrate_x_ =( flowrate_x_.abs() + 0.000001)  ** (1 / 1.852) *  torch.sign(flowrate_x_)
        # # head_loss_edge_attr_ = self.get_headloss_from_flowrates(batch.edge_y, batch).squeeze()
        # assert torch.allclose(flowrate_x_[virtual], batch.edge_y[virtual], atol=1e-3)

        target = output['edge_attr'] if self.target == 'edge_attr' else batch.edge_y
        return ((flowrate_x - target.squeeze()[virtual]).abs() ** self.p).mean()

    def inverse_transform_heads(self, x):
        for transform in self.transform:
            if isinstance(transform, MinMaxNormalize):
                break

        max = transform.max_value['x'][self.heads_index]
        min = transform.min_value['x'][self.heads_index]

        x = x * (max - min) + min
        return x


class HeadLossMSELoss(HeadLoss):

    def forward(self, output, batch):
        out = super(HeadLossMSELoss, self).forward(output, batch)
        return ((out[0] - out[1]).abs() ** self.p).mean()

    def __str__(self):
        return 'HeadLossMSE'


class HeadLossNorm(HeadLoss):

    def forward(self, output, batch):
        out = super().forward(output, batch)
        return (out[0].abs() ** self.p).mean()

    def __str__(self):
        return 'HeadLossNorm'


class IntermediatePredictionLoss(BaseLoss):
    def __init__(self, key='intermediate_x', weight=1, p=2, discounting=0.9, *args, **kwargs):
        self.discounting = discounting
        self.key = key
        self.p = p
        # self.loss_func = torch.nn.MSELoss(reduction='none')

        super().__init__(weight, **kwargs)

    def forward(self, output, batch):
        output = output[self.key]

        y = batch.node_y if 'node_y' in batch.keys else batch.y
        target = y.unsqueeze(-1).repeat((1, output.shape[1])).unsqueeze(-1)
        layers = target.shape[1]

        loss = (output - target) ** self.p if self.p != 1 else torch.abs(output - target)
        base = ((torch.ones(layers) * self.discounting) ** torch.arange(1, layers + 1).flip(0)).squeeze()

        loss = loss * base.to(loss.device)
        loss = torch.sum(loss, dim=1)
        loss = torch.mean(loss)
        return loss

    def __str__(self):
        return 'IntermediatePredictionLoss'


class HarmonicFlowSmoothing(BaseLoss):

    def __init__(self, dataset, weight=1, p=2, operator='boundary',
                 intermediate_loss= False, derived_flowrates = False,
                 **kwargs):
        data = dataset.datasets[0].data
        self.p = p
        self.intermediate_loss = intermediate_loss

        assert operator in ['boundary', 'laplacian']

        self.operator = operator
        self.derived_flowrates = derived_flowrates

        self.virtual_node_index = data.x_names.index('virtual') if 'virtual' in data.x_names else None
        self.virtal_edge_index = data.edge_attr_names.index('virtual') if 'virtual' in data.edge_attr_names else None
        self.demand_idx = data.x_names.index('demand') if 'demand' in data.x_names else None
        self.junction_idx = data.x_names.index('Junction') if 'Junction' in data.x_names else None

        self.heads_idx = data.x_names.index('head') if 'head' in data.x_names else None
        self.loss_coefficient_index = data.edge_attr_names.index(
            'loss_coefficient') if 'loss_coefficient' in data.edge_attr_names else None
        self.transform = dataset.datasets[0].pre_transform

        super().__init__(weight, **kwargs)

    @profile
    def forward(self, output, batch, **kwargs):

        loss_coefficient = torch.abs(batch.edge_attr[:, self.loss_coefficient_index])

        if not self.derived_flowrates:
            x = output['edge_attr']
            output_ = batch.edge_y

            # simplices = [(s.tolist(), 0.0) for s in batch.edge_index.T]
            # simplices += [([n], 0.0) for n in range(batch.num_nodes)]
            # hl = HodgeLaplacians(simplices, mode='gudhi', maxdimension=1)
            # B1 = hl.getBoundaryOperator(1)
            # B1 = scipy.sparse.coo_matrix(B1)
            #
            # boundary_index = torch.tensor(np.vstack((B1.row, B1.col)), dtype=torch.long, device=output.device)
            # boundary_weight = torch.tensor(B1.data, dtype=torch.float, device=output.device)
            # assert torch.all(boundary_index == batch.boundary_index)
            # assert torch.all(boundary_weight == batch.boundary_weight)

            # B1 = torch.sparse_coo_tensor(batch.boundary_index, batch.boundary_weight,
            #                              size=(batch.num_nodes, batch.num_edges))
            # check = torch.sparse.mm(B1, batch.y.unsqueeze(-1).to_sparse()).to_dense()
        else:
            if 'intermediate_x' in output and self.intermediate_loss:
                heads = output['intermediate_x']
            else:
                heads = output['x'].unsqueeze(-1)

            heads = self.inverse_transform_heads(heads)

            x = self.get_flowrates_from_nodal_heads(batch, heads, loss_coefficient)

            # run checks
            y = batch.node_y if 'node_y' in batch.keys else batch.y
            heads = y.unsqueeze(-1).unsqueeze(-1)
            # heads = self.inverse_transform_heads(heads)
            output_ = self.get_flowrates_from_nodal_heads(batch, heads, loss_coefficient)

        dim_size = batch.edge_index.max() + 1

        # get virtual indices
        if self.virtual_node_index is not None:
            virtual_indices = batch.x[:, self.virtual_node_index] == 0
        elif self.virtal_edge_index is not None:
            virtual_indices = batch.edge_attr[:, self.virtal_edge_index] == 0
        else:
            if self.operator == 'boundary':
                virtual_indices = torch.ones(batch.num_nodes, dtype=torch.bool, device=x.device)
            else:
                virtual_indices = torch.ones(batch.num_edges, dtype=torch.bool, device=x.device)

        check = scatter(output_, batch.edge_index[0], 0, dim_size=dim_size) - scatter(output_, batch.edge_index[1],
                                                                                      0,
                                                                                      dim_size=dim_size)
        check = check[virtual_indices].squeeze()

        error = scatter(x, batch.edge_index[0], 0, dim_size=dim_size) - scatter(x, batch.edge_index[1], 0,
                                                                                      dim_size=dim_size)
        error = error[virtual_indices].squeeze(-1)
        # error = operator.matmul(output.squeeze(-1))
        # error = error[virtual_indices]

        if self.virtal_edge_index is None:
            demands = batch.x[:, self.demand_idx]
            check = check.squeeze() - demands
            error = error - demands.unsqueeze(-1).repeat(1,error.shape[-1])
            # error = error.squeeze() - demands.unsqueeze(-1).expand(demands.shape[0], error.shape[1])

        if not torch.allclose(check, torch.tensor(0, dtype=torch.float), atol=1e-3):
            amax = check.abs().argmax()
            # slice_idx = -(batch.slices['edge_attr'] > amax).sum()
            # amax = check.abs().argmax()
            # wds_name = batch[slice_idx].wds_names
            print(
                'Flow conservation violated. amax:{},max:{}, network_name:'.format(amax, check.abs().max()))

        # calculate discounted loss if applicable
        if error.squeeze().dim() > 1:
            layers = error.shape[1]
            discounting = 0.9
            base = ((torch.ones(layers) * discounting) ** torch.arange(1, layers + 1).flip(0)).squeeze()
            error = error ** self.p
            error = error * base.to(error.device)
            error =  error.sum(dim=1)
        else:
            error = (error.abs() ** self.p)

        # # filter junction nodes
        # if self.junction_idx is not None:
        #     junction_indices = batch.x[:, self.junction_idx] == 1
        #     error = error[junction_indices]

        return error.mean()

    @profile
    def get_flowrates_from_nodal_heads(self, batch, heads, loss_coefficient):
        head_loss_x = get_headloss_from_nodes(batch, heads)
        flowrate_x = (head_loss_x / loss_coefficient.view(-1, 1, 1))
        flowrate_abs = torch.abs(flowrate_x)
        if torch.any(torch.isnan(flowrate_abs)) or torch.any(torch.isinf(flowrate_abs)) or torch.any(flowrate_abs < 0):
            print('nan')
        flowrate_x = (flowrate_x.abs() + 0.0000001) ** (1 / 1.852) * torch.sign(flowrate_x)
        return flowrate_x

    def get_operator(self, batch, output):
        if self.operator == 'boundary':
            operator = SparseTensor(row=batch.boundary_index[0],
                                    col=batch.boundary_index[1],
                                    value=batch.boundary_weight,
                                    sparse_sizes=(batch.num_nodes, batch.num_edges))

        elif self.operator == 'laplacian':
            operator = SparseTensor(row=batch.laplacian_index[0],
                                    col=batch.laplacian_index[1],
                                    value=batch.laplacian_weight,
                                    sparse_sizes=(batch.num_edges, batch.num_edges))
        return operator



    def inverse_transform_heads(self, x):
        for transform in self.transform:
            if isinstance(transform, MinMaxNormalize):

                max = transform.max_value['x'][self.heads_idx]
                min = transform.min_value['x'][self.heads_idx]

                x = x * (max - min) + min
                break
        return x

    def __str__(self):
        return 'HarmonicFlowSmoothing'

class HarmonicFlowSmoothingGather(HarmonicFlowSmoothing):

    def get_flowrates_from_nodal_heads(self, batch, heads, loss_coefficient):
        H_i = torch.gather(heads, 0, batch.edge_index[0].unsqueeze(-1).repeat(1, heads.shape[1]).unsqueeze(-1))
        H_j = torch.gather(heads, 0, batch.edge_index[1].unsqueeze(-1).repeat(1, heads.shape[1]).unsqueeze(-1))
        head_loss_x = H_i - H_j
        flowrate_x = (head_loss_x / loss_coefficient.view(-1, 1, 1))
        flowrate_abs = torch.abs(flowrate_x)
        flowrate_x = (flowrate_x.abs() + 0.0000001) ** (1 / 1.852) * torch.sign(flowrate_x)
        return flowrate_x

    def __str__(self):
        return 'HarmonicFlowSmoothingGather'

class HarmonicFlowSmoothingDense(torch.nn.Module):

    # @profile
    def forward(self, output, batch, **kwargs):
        output = output['edge_attr'] if 'edge_attr' in output else output

        B1 = torch.sparse_coo_tensor(batch.boundary_index, batch.boundary_weight,
                                     size=(batch.num_nodes, batch.num_edges))
        # B1 = torch.sparse_coo_tensor(batch.laplacian_index, batch.laplacian_weight, size=(batch.num_edges, batch.num_edges))

        check = torch.sparse.mm(B1, batch.y.unsqueeze(-1).to_sparse()).to_dense()
        if not torch.allclose(check, torch.tensor(0, dtype=torch.float), atol=1e-6):
            amax = check.abs().argmax()
            # slice_idx = -(batch.slices['edge_attr'] > amax).sum()
            # amax = check.abs().argmax()
            # wds_name = batch[slice_idx].wds_names
            print(
                'Flow conservation violated. amax:{},max:{}, network_name:'.format(amax, check.abs().max()))

        out = torch.mean(torch.sparse.mm(B1, output.to_sparse()).to_dense() ** 2)
        return out

    def __str__(self):
        return 'HarmonicFlowSmoothing'


class CompositeLossFunction(torch.nn.Module, IterableCompose):
    def __init__(self, losses: List[BaseLoss]):
        super().__init__()
        self.loss_functions = losses

    def forward(self, output, batch):
        return CompositeLoss(
            [Loss(loss(output, batch),
                  str(loss),
                  loss.no_grad) for loss in self.loss_functions])

    @property
    def iterable(self):
        return self.loss_functions


@dataclass
class Loss:
    loss: Tensor
    __str__: str
    no_grad: bool

    def item(self):
        return self.loss

    def __str__(self):
        return self.__str__

    def no_grad(self):
        return self.no_grad

    def __mul__(self, other):
        self.loss *= other
        return self


@dataclass
class CompositeLoss(Loss):
    loss: List[Loss]

    def __str__(self):
        return ' + '.join([str(loss) for loss in self.loss])

    def item(self):
        return sum([loss.item() for loss in self.loss if not loss.no_grad])

    @property
    def iterable(self):
        return self.loss

    def __iter__(self):
        return Iterator(self)

    def __getitem__(self, index):
        return self.iterable[index]

    def __mul__(self, other):
        return CompositeLoss([loss * other for loss in self.loss])


@profile
def get_headloss_from_nodes(batch, heads):
    # get headloss from labels
    from_nodes = batch.edge_index[0]
    to_nodes = batch.edge_index[1]
    # heads = batch.x[:, self.heads_index]

    y_head_diff = (heads[to_nodes] - heads[from_nodes])
    return y_head_diff





class IntermediateLoss:

    def __init__(self, layers, discount_factor):
        self.base = ((torch.ones(layers) * discount_factor) ** torch.arange(1, layers + 1).flip(0)).unsqueeze(-1)

    def __call__(self, errors):
        errors = (errors * self.base)
        errors = torch.sum(errors, dim=1)
        return errors
