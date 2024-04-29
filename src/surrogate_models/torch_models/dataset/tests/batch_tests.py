import torch

from src.surrogate_models.torch_models.data.complex_data import ComplexBatch
from src.surrogate_models.torch_models.data.data import GraphData


def test_headloss(data_loader):
    test_results = []
    for batch in data_loader:
        if isinstance(batch, GraphData):
            heads = batch.y
            from_nodes = batch.edge_index[0]
            to_nodes = batch.edge_index[1]
        elif isinstance(batch, ComplexBatch):
            heads = batch.cochains[0].y
            from_nodes = batch.cochains[0].upper_index[0, ::2]
            to_nodes = batch.cochains[0].upper_index[1, ::2]
        else:
            raise TypeError('data type is incorrect')

        for transform in data_loader.dataset.datasets[0].pre_transform:
            if hasattr(transform, 'scale') or hasattr(transform, 'inverse'):
                batch = transform(batch, inverse=True)

        if 'flowrate' in data_loader.dataset.datasets[0].data.edge_attr_names:
            flowrate = batch.edge_attr[:,
                       data_loader.datasets[0].data.edge_attr_names.index('flowrate')]
            loss_coefficient = batch.edge_attr[:,
                               data_loader.datasets[0].data.edge_attr_names.index('loss_coefficient')]

            flowrate = (flowrate.sign() * (flowrate.abs() ** 1.852)) * loss_coefficient


        hl = heads[to_nodes] - heads[from_nodes]
        errs = torch.abs(hl - flowrate.view(hl.shape))

        if 'virtual' in data_loader.dataset.datasets[0].data.edge_attr_names:
            virtual = batch.edge_attr[:,
                      data_loader.datasets[0].data.edge_attr_names.index('virtual')]
            errs = errs * torch.logical_not(virtual)
        err_max = torch.max(errs)
        # assert err < 1e-3
        print(err_max < 1e-3)
