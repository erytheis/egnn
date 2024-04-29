import warnings
from typing import List, Optional, Mapping

import numpy as np
import torch
from line_profiler_pycharm import profile
from torch import Tensor
from torch_geometric.data import Data

from src.surrogate_models.torch_models.dataset.transforms.base import BaseTransform


class MinMaxNormalize(BaseTransform):
    """Normalize a tensor with min and max values.
    """
    keys: List = ['x', 'y', 'edge_attr']

    def __init__(self, keys=None,
                 min_value: Optional[Mapping[str, float]] = None,
                 max_value: Optional[Mapping[str, float]] = None,
                 columns: Optional[List[str]] = None,
                 columns_idx: Optional[Mapping[str, int]] = None,
                 inplace: bool = False):
        super().__init__()

        self.min_value = min_value or {}
        self.max_value = max_value or {}
        self.columns_idx = columns_idx or {}
        self.ignore_columns_idx = {k: [] for k in self.keys}
        self.inplace = inplace
        self.columns = columns
        if keys is not None:
            self.keys = keys
        if columns is None:
            self.columns = ["demand", "head", "elevation", "pressure", "length", "roughness", "diameter"]
        super().__init__()

    def scale(self):
        pass

    # @profile
    def forward(self, data: Data, inverse=False) -> Data:
        for key in self.min_value.keys():
            if isinstance(data[key], torch.Tensor):
                old = data[key].clone()
                data[key] = self.minmaxnormalize(data[key],
                                                 self.min_value[key],
                                                 self.max_value[key],
                                                 self.inplace,
                                                 inverse=inverse)
                # test that the scaling is consistent
                if not inverse:
                    new = self.minmaxnormalize(data[key],
                                               self.min_value[key],
                                               self.max_value[key],
                                               False,
                                               inverse=True)
                    assert torch.allclose(old, new, atol=1e-5)
        return data

    def transform_tensor(self, tensor: Tensor, key: str, inverse=False) -> Tensor:
        return self.minmaxnormalize(tensor,
                                    self.min_value[key],
                                    self.max_value[key],
                                    self.inplace,
                                    inverse=inverse)

    def _infer_parameters(self, data: Data) -> None:
        for key in self.keys:
            if f'{key}_names' not in data:
                continue

            # names = getattr(data, f'{key}_names')
            names = data[f'{key}_names']

            self.columns_idx[key] = []

            for col in self.columns:
                if col in names:
                    self.columns_idx[key].append(names.index(col))

            # create a list of indices "except" the ones to normalize
            self.ignore_columns_idx[key] = list(range(len(names)))
            [self.ignore_columns_idx[key].remove(i) for i in self.columns_idx[key]]

            if data[key] is None:
                continue

            self.min_value[key] = data[key].min(dim=0).values
            self.max_value[key] = data[key].max(dim=0).values
            # check if min_value equals max_value
            if torch.any(self.min_value[key] == self.max_value[key]):
                warnings.warn(f"Min and max values are equal for {key}")
                equal_idx = self.min_value[key] == self.max_value[key]
                self.min_value[key][equal_idx] = 0
                self.max_value[key][equal_idx] = 1

            # Mask min_values and max_values that won't be normalized
            if len(self.ignore_columns_idx[key]) > 0:
                if self.min_value[key].ndim > 0:
                    self.min_value[key][self.ignore_columns_idx[key]] = torch.tensor(0, dtype=torch.float)
                    self.max_value[key][self.ignore_columns_idx[key]] = torch.tensor(1, dtype=torch.float)
                else:
                    self.min_value[key] = 0
                    self.max_value[key] = 1

    def __repr__(self) -> str:
        return '{}(min={}, max={})'.format(self.__class__.__name__, self.min_value, self.max_value)

    @staticmethod
    def minmaxnormalize(tensor: Tensor,
                        min_value: float,
                        max_value: float,
                        inplace: bool = False,
                        inverse=False) -> Tensor:
        """Normalize a float tensor with min and max value.

        .. note::
            This transform acts out of place by default, i.e., it does not mutates the input tensor.

        See :class:`~torchvision.transforms.Normalize` for more details.

        Returns:
            Tensor: Normalized Tensor .
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

        if not tensor.is_floating_point():
            raise TypeError('Input tensor should be a float tensor. Got {}.'.format(tensor.dtype))

        if not inplace:
            tensor = tensor.clone()

        if min == max:
            raise ValueError('min and max are equal, cannot normalize.')

        dtype = tensor.dtype
        min_value = torch.as_tensor(min_value, dtype=dtype, device=tensor.device)
        max_value = torch.as_tensor(max_value, dtype=dtype, device=tensor.device)
        if inverse:
            return tensor.mul_(max_value - min_value).add_(min_value)
        else:
            return tensor.sub_(min_value).div_(max_value - min_value)


class Standartize(BaseTransform):
    keys: List = ['x', 'y', 'edge_attr']

    def __init__(self, keys=None,
                 mean=None, std=None,
                 columns: Optional[List[str]] = None,
                 columns_idx: Optional[List[int]] = None,
                 inplace=False):

        self.mean = mean or {}
        self.std = std or {}
        self.inplace = inplace
        self.columns = columns
        self.columns_idx = columns_idx or {}
        self.ignore_columns_idx = {k: [] for k in self.keys}
        if keys is not None:
            self.keys = keys
        if columns is None:
            self.columns = ["demand", "head", "elevation", "pressure", "length", "roughness", "diameter"]
        super().__init__()

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean or 0, self.std or 1)

    def forward(self, data: Data, inverse=False) -> Data:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        for key in self.mean.keys():
            if isinstance(data[key], torch.Tensor):
                for key in self.mean.keys():
                    data[key] = self.standartize(data[key],
                                                 self.mean[key],
                                                 self.std[key],
                                                 self.inplace,
                                                 inverse=inverse)
        return data

    @torch.no_grad()
    def _infer_parameters(self, data):
        for key in self.keys:
            if f'{key}_names' not in data:
                continue
            # names = getattr(data, f'{key}_names')
            names = data[f'{key}_names']

            self.columns_idx[key] = []

            for col in self.columns:
                if col in names:
                    self.columns_idx[key].append(names.index(col))

            # create a list of indices "except" the ones to normalize
            self.ignore_columns_idx[key] = list(range(len(names)))
            [self.ignore_columns_idx[key].remove(i) for i in self.columns_idx[key]]

            if data[key] is None:
                continue

            self.mean[key] = data[key].mean(axis=0)
            self.std[key] = data[key].std(axis=0)

            # check if std equals 0
            if torch.any(self.std[key] == 0):
                warnings.warn(f"std is zero for {key}. Setting up to 1")
                equal_idx = self.std[key] == 0
                self.mean[key][equal_idx] = 0
                self.std[key][equal_idx] = 1

            # Mask attributes that won't be normalized
            if len(self.ignore_columns_idx[key]) > 0:
                if self.mean[key].ndim > 0:
                    self.mean[key][self.ignore_columns_idx[key]] = torch.tensor(0, dtype=torch.float)
                    self.std[key][self.ignore_columns_idx[key]] = torch.tensor(1, dtype=torch.float)
                else:
                    self.mean[key] = 0
                    self.std[key] = 1

    @staticmethod
    def standartize(tensor: Tensor,
                    mean: List[float],
                    std: List[float],
                    inplace: bool = False,
                    inverse: bool = False) -> Tensor:
        """Normalize a float tensor with mean and standard deviation.

        .. note::
            This transform acts out of place by default, i.e., it does not mutates the input tensor.

        See :class:`~torchvision.transforms.Normalize` for more details.

        Returns:
            Tensor: Normalized Tensor .
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

        if not tensor.is_floating_point():
            raise TypeError('Input tensor should be a float tensor. Got {}.'.format(tensor.dtype))

        if not inplace:
            tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = mean.view(1, -1)
        if std.ndim == 1:
            std = std.view(1, -1)

        # inverse standardization by multiplying and adding
        if inverse:
            return tensor.mul_(std).add_(mean)
        else:
            return tensor.sub_(mean).div_(std)


class Scale(BaseTransform):
    keys: List = ['x', 'y', 'edge_attr']

    def __init__(self, value, columns=None, attribute_key=None, columns_idx=None, extend_dimensions=False, ):
        super().__init__()
        self.value = value
        self.columns = columns
        self.attribute_key = attribute_key
        self.columns_idx = columns_idx or {}
        self.ignore_columns_idx = {k: [] for k in self.keys}
        self.extend_dimensions = extend_dimensions
        self.scaler = {}

    def _scale(self, tensor, key, inverse):
        if not inverse:
            scaler = self.scaler[key].to(tensor.device) if hasattr(self.scaler[key], 'to') else self.scaler[key]
        else:
            scaler = 1 / self.scaler[key].to(tensor.device) if hasattr(self.scaler[key], 'to') else 1 / self.scaler[key]
        return tensor * scaler

    def forward(self, data, inverse=False, inplace=True, *args, **kwargs):
        for key in self.columns_idx.keys():
            if isinstance(data[key], torch.Tensor):

                if not inplace:
                    tensor = data[key].clone()
                else:
                    tensor = data[key]

                data[key] = self._scale(tensor, key, inverse=inverse)

        return data

    def _infer_parameters(self, data, *args, **kwargs):
        for key in self.keys:
            if f'{key}_names' not in data:
                continue

            if data[key] is None:
                continue

            if not any([col in data[f'{key}_names'] for col in self.columns]):
                continue

            # get names
            names = data[f'{key}_names']

            if self.extend_dimensions:
                attribute = data[key]
                new_attributes = torch.zeros(
                    (len(attribute), len([c for c in self.columns if c in names])), dtype=attribute.dtype,
                    device=attribute.device
                )
                for i, col in enumerate(self.columns):
                    if col in names:
                        new_attributes[:, i] = attribute[:, names.index(col)]
                attribute = torch.cat([attribute, new_attributes], dim=-1)

                [data['{}_names'.format(key)].append(f'{col}_scaled') for col in self.columns if col in names]
                data[key] = attribute

            self.columns_idx[key] = []
            # for col in self.columns:
            #     if col in names:
            #         if self.extend_dimensions:
            #             col = f'{col}_scaled'
            #         self.columns_idx[key].append(names.index(col))

            if self.scaler is not None:
                self.scaler[key] = torch.ones(len(names), device=data[key].device) if data[key].ndim > 1 else self.value

            for col in self.columns:
                if col in names:
                    if self.extend_dimensions:
                        col = f'{col}_scaled'
                    scale_idx = names.index(col)
                    self.columns_idx[key].append(scale_idx)

                    if self.scaler is not None:
                        self.scaler[key][scale_idx] = self.value


class Log(Scale):
    loss_coefficient_index = 1

    def __init__(self, base=None, columns=None, attribute_key=None, columns_idx=None, extend_dimensions=False, ):
        super().__init__(base, columns, attribute_key, columns_idx, extend_dimensions)
        self.scaler = None

    def _scale(self, tensor, key, inverse):

        for i in self.columns_idx[key]:
            t = tensor[:, i]

            if self.value is None:
                t = torch.log(t)
            elif self.value == 10:
                t = torch.log10(t)

            t[t == -np.inf] = t[t != -np.inf].min()
            t[t == np.inf] = t[t != np.inf].max()

            tensor[:, i] = t
        return tensor

    def inverse(self, batch, inplace=False):
        edge_attr = batch.edge_attr
        if not inplace:
            edge_attr = batch.edge_attr.clone()

        if self.value is None:
            edge_attr[:, self.loss_coefficient_index] = torch.exp(edge_attr[:, self.loss_coefficient_index])
        elif self.value == 10:
            edge_attr[:, self.loss_coefficient_index] = torch.pow(10, edge_attr[:, self.loss_coefficient_index])

        batch.edge_attr = edge_attr
        return batch

    def forward(self, batch, inverse=False, *args, **kwargs):
        if not inverse:
            return super().forward(batch, inverse=inverse, *args, **kwargs)
        else:
            return self.inverse(batch)


class ScaleOutflowsToTotal(Scale):
    def __init__(self, columns=None, attribute_key='flowrate', columns_idx=None, reference_key='Reservoir',
                 reference_idx=None, ):
        self.reference_key = reference_key
        self.reference_idx = reference_idx
        self.attribute_key = attribute_key

        super().__init__(1, columns, attribute_key, columns_idx)

    def forward(self, data, *args, **kwargs):
        value = data.x[:, self.reference_idx].sum()
        data.edge_attr[:, self.attribute_idx] /= value
        return data

    def _infer_parameters(self, data, *args, **kwargs):
        if self.reference_idx is None:
            self.reference_idx = data.x_names.index(self.reference_key)

        self.attribute_idx = data.edge_attr_names.index(self.attribute_key)


class Power(Scale):

    def _scale(self, tensor, key, inverse):
        if not inverse:
            scaler = self.scaler[key].to(tensor.device) if hasattr(self.scaler[key], 'to') else self.scaler[key]
        else:
            scaler = 1 / self.scaler[key].to(tensor.device) if hasattr(self.scaler[key], 'to') else 1 / self.scaler[key]
        return tensor.sign() * (tensor.abs() ** scaler)
